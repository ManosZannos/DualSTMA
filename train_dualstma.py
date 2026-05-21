"""
train_dualstma.py

Training script for DualSTMA v3 (Huang et al., JMSE 2024).

v3 changes:
  - surr_mask: boolean mask for real vs padded surrounding vessels
  - Passed to model for key_padding_mask in spatial attention (Eq. 13, 18)
  - Removed AMP (autocast/GradScaler) — paper uses float32 on RTX 3090Ti

Usage (RTX 4090):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5

Usage (ihatz A100):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5 --gpu_num 0
"""

import os
import sys
import time
import math
import argparse
import pickle
import glob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import pandas as pd

from model_dualstma import DualSTMA, rotate_translate

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num',    default="",   type=str)
parser.add_argument('--dataset',    default='marinecadastre_2021')
parser.add_argument('--obs_len',    type=int, default=10)
parser.add_argument('--pred_len',   type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr',         type=float, default=0.0015)
parser.add_argument('--tag',        default='DualSTMA_v3')
parser.add_argument('--resume',     action='store_true', default=False)
parser.add_argument('--max_surr',   type=int, default=10)

args_early, _ = parser.parse_known_args()
if args_early.gpu_num:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_early.gpu_num
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Training DualSTMA v3 (float32, no AMP) ...')
print(args)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DualSTMADataset(Dataset):
    """
    DualSTMA dataset with vessel-centered rotation transform.
    v3: surr_mask — True=real vessel, False=padded zero
    """

    def __init__(self, data_dir, obs_len=10, pred_len=5, max_surr=10):
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.max_surr = max_surr
        self.seq_len  = obs_len + pred_len

        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if not csv_files:
            raise RuntimeError(f'No CSV files found in {data_dir}')

        dfs = []
        for f in csv_files:
            df = pd.read_csv(f)
            print(f'  {os.path.basename(f)}: {df["vessel_id"].nunique()} vessels')
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.sort_values(['frame_id', 'vessel_id']).reset_index(drop=True)

        self.samples = self._build_samples()
        print(f'  Total samples: {len(self.samples):,}')

    def _build_samples(self):
        samples = []
        for vessel_id, group in self.df.groupby('vessel_id'):
            group = group.sort_values('frame_id').reset_index(drop=True)
            n = len(group)
            if n < self.seq_len:
                continue
            for start in range(n - self.seq_len + 1):
                frame_ids = group.iloc[start:start + self.seq_len]['frame_id'].values
                if np.all(np.diff(frame_ids) == 1):
                    samples.append((vessel_id, start))
        return samples

    def _apply_rotation(self, lon, lat, heading_rad_tn):
        cos_h = np.cos(-heading_rad_tn)
        sin_h = np.sin(-heading_rad_tn)
        lon_r =  cos_h * lon + sin_h * lat
        lat_r = -sin_h * lon + cos_h * lat
        return lon_r, lat_r

    def _get_dynamic_features(self, rows, heading_rad_tn, origin_lon, origin_lat):
        """8-channel dynamic features after vessel-centered rotation."""
        lon     = rows['LON'].values.astype(np.float32)
        lat     = rows['LAT'].values.astype(np.float32)
        sog     = rows['SOG'].values.astype(np.float32)
        heading = rows['Heading'].values.astype(np.float32)

        lon_t = lon - origin_lon
        lat_t = lat - origin_lat
        lon_r, lat_r = self._apply_rotation(lon_t, lat_t, heading_rad_tn)

        dlon_r = np.zeros_like(lon_r)
        dlat_r = np.zeros_like(lat_r)
        dlon_r[1:] = lon_r[1:] - lon_r[:-1]
        dlat_r[1:] = lat_r[1:] - lat_r[:-1]

        heading_rad = heading * 2 * np.pi
        v_lon_orig = sog * np.sin(heading_rad)
        v_lat_orig = sog * np.cos(heading_rad)

        cos_h = np.cos(-heading_rad_tn)
        sin_h = np.sin(-heading_rad_tn)
        v_lon_r =  cos_h * v_lon_orig + sin_h * v_lat_orig
        v_lat_r = -sin_h * v_lon_orig + cos_h * v_lat_orig

        theta_lon_orig = np.sin(heading_rad)
        theta_lat_orig = np.cos(heading_rad)
        theta_lon_r =  cos_h * theta_lon_orig + sin_h * theta_lat_orig
        theta_lat_r = -sin_h * theta_lon_orig + cos_h * theta_lat_orig

        return np.stack([
            lon_r, lat_r, dlon_r, dlat_r,
            v_lon_r, v_lat_r, theta_lon_r, theta_lat_r
        ], axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vessel_id, start = self.samples[idx]

        vessel_rows = self.df[self.df['vessel_id'] == vessel_id].sort_values('frame_id').reset_index(drop=True)
        window      = vessel_rows.iloc[start:start + self.seq_len]
        obs_rows    = window.iloc[:self.obs_len]
        pred_rows   = window.iloc[self.obs_len:self.obs_len + self.pred_len]

        assert len(pred_rows) == self.pred_len

        origin_lon     = float(obs_rows['LON'].iloc[-1])
        origin_lat     = float(obs_rows['LAT'].iloc[-1])
        heading_norm   = float(obs_rows['Heading'].iloc[-1])
        heading_rad_tn = heading_norm * 2 * np.pi

        all_features = self._get_dynamic_features(
            window, heading_rad_tn, origin_lon, origin_lat
        )
        obs_features = all_features[:self.obs_len]

        v_type   = int(obs_rows['vessel_type'].iloc[0])
        v_width  = int(obs_rows['vessel_width'].iloc[0])
        v_length = int(obs_rows['vessel_length'].iloc[0])

        gt_lon = pred_rows['LON'].values.astype(np.float32)
        gt_lat = pred_rows['LAT'].values.astype(np.float32)
        gt_pos = np.stack([gt_lon, gt_lat], axis=-1)

        pred_sog          = pred_rows['SOG'].values.astype(np.float32)
        pred_heading_norm = pred_rows['Heading'].values.astype(np.float32)
        pred_heading_rad  = pred_heading_norm * 2 * np.pi

        v_lon_orig = pred_sog * np.sin(pred_heading_rad)
        v_lat_orig = pred_sog * np.cos(pred_heading_rad)
        gt_vel = np.stack([v_lon_orig, v_lat_orig], axis=-1)

        theta_lon_orig = np.sin(pred_heading_rad)
        theta_lat_orig = np.cos(pred_heading_rad)
        gt_heading = np.stack([theta_lon_orig, theta_lat_orig], axis=-1)

        # Surrounding vessels
        obs_frame_ids   = obs_rows['frame_id'].values
        frame_start_obs = obs_frame_ids[0]
        frame_end_obs   = obs_frame_ids[-1]

        surr_df      = self.df[
            (self.df['vessel_id'] != vessel_id) &
            (self.df['frame_id'] >= frame_start_obs) &
            (self.df['frame_id'] <= frame_end_obs)
        ]
        surr_vessels = surr_df['vessel_id'].unique()

        surr_features_list = []
        surr_types   = []
        surr_widths  = []
        surr_lengths = []

        for sv in surr_vessels[:self.max_surr]:
            sv_rows   = surr_df[surr_df['vessel_id'] == sv].sort_values('frame_id')
            sv_window = sv_rows[sv_rows['frame_id'].isin(obs_frame_ids)].sort_values('frame_id')
            if len(sv_window) != self.obs_len:
                continue
            sv_feat = self._get_dynamic_features(
                sv_window, heading_rad_tn, origin_lon, origin_lat
            )
            surr_features_list.append(sv_feat)
            surr_types.append(int(sv_window['vessel_type'].iloc[0]))
            surr_widths.append(int(sv_window['vessel_width'].iloc[0]))
            surr_lengths.append(int(sv_window['vessel_length'].iloc[0]))

        n_real = len(surr_features_list)

        surr_dynamic = np.zeros((self.max_surr, self.obs_len, 8), dtype=np.float32)
        s_types      = np.zeros(self.max_surr, dtype=np.int64)
        s_widths     = np.zeros(self.max_surr, dtype=np.int64)
        s_lengths    = np.zeros(self.max_surr, dtype=np.int64)
        surr_mask    = np.zeros(self.max_surr, dtype=bool)

        for i, feat in enumerate(surr_features_list):
            surr_dynamic[i] = feat
            s_types[i]      = surr_types[i]
            s_widths[i]     = surr_widths[i]
            s_lengths[i]    = surr_lengths[i]
            surr_mask[i]    = True

        return {
            'obs_features':  torch.tensor(obs_features,  dtype=torch.float32),
            'gt_pos':        torch.tensor(gt_pos,         dtype=torch.float32),
            'gt_vel':        torch.tensor(gt_vel,         dtype=torch.float32),
            'gt_heading':    torch.tensor(gt_heading,     dtype=torch.float32),
            'last_obs_pos':  torch.tensor([origin_lon, origin_lat], dtype=torch.float32),
            'heading_rad':   torch.tensor(heading_rad_tn, dtype=torch.float32),
            'v_type':        torch.tensor(v_type,   dtype=torch.long),
            'v_width':       torch.tensor(v_width,  dtype=torch.long),
            'v_length':      torch.tensor(v_length, dtype=torch.long),
            'surr_dynamic':  torch.tensor(surr_dynamic, dtype=torch.float32),
            's_types':       torch.tensor(s_types,   dtype=torch.long),
            's_widths':      torch.tensor(s_widths,  dtype=torch.long),
            's_lengths':     torch.tensor(s_lengths, dtype=torch.long),
            'surr_mask':     torch.tensor(surr_mask, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def wrapped_heading_error_floormod(pred_h, gt_h):
    """FIX #11: Wrapped heading error with floormod (Eq. 38)."""
    diff = pred_h - gt_h
    return -math.pi + torch.fmod(diff + math.pi, 2 * math.pi)


def dualstma_loss(pred_pos, pred_vel, pred_heading,
                  gt_pos, gt_vel, gt_heading,
                  lambda_pos=10.0, lambda_heading=1.0, lambda_vel=0.1):
    """DualSTMA loss (Eq. 36-41). All in original coordinate space."""
    pred_len = pred_pos.shape[1]

    weights = torch.tensor(
        [(pred_len - t) for t in range(pred_len)],
        dtype=torch.float32, device=pred_pos.device
    )
    weights = weights / weights.sum()

    pos_loss = sum(
        weights[t] * F.huber_loss(pred_pos[:, t], gt_pos[:, t])
        for t in range(pred_len)
    )
    vel_loss     = F.huber_loss(pred_vel, gt_vel)
    h_error      = wrapped_heading_error_floormod(pred_heading, gt_heading)
    heading_loss = F.huber_loss(h_error, torch.zeros_like(h_error))

    total = lambda_pos * pos_loss + lambda_heading * heading_loss + lambda_vel * vel_loss
    return total, pos_loss, vel_loss, heading_loss


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {
    'min_val_epoch': -1, 'min_val_loss': 9999999999,
    'min_train_epoch': -1, 'min_train_loss': 9999999999
}


def run_epoch(model, loader, optimizer, checkpoint_dir, epoch, train=True):
    global metrics, constant_metrics
    model.train() if train else model.eval()

    total_loss = 0.0
    n_batches  = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            obs        = batch['obs_features'].to(device)
            gt_pos     = batch['gt_pos'].to(device)
            gt_vel     = batch['gt_vel'].to(device)
            gt_heading = batch['gt_heading'].to(device)
            last_obs   = batch['last_obs_pos'].to(device)
            head_rad   = batch['heading_rad'].to(device)

            v_type   = batch['v_type'].to(device)
            v_width  = batch['v_width'].to(device)
            v_length = batch['v_length'].to(device)

            surr_dyn  = batch['surr_dynamic'].to(device)
            s_types   = batch['s_types'].to(device)
            s_widths  = batch['s_widths'].to(device)
            s_lengths = batch['s_lengths'].to(device)
            surr_mask = batch['surr_mask'].to(device)

            target_static      = (v_type, v_width, v_length)
            surrounding_static = (s_types, s_widths, s_lengths)

            # Forward pass — float32, no AMP
            pred_pos, pred_vel, pred_heading = model(
                obs, target_static,
                head_rad, last_obs,
                surr_dyn, surrounding_static,
                surr_mask=surr_mask
            )

            loss, pos_l, vel_l, head_l = dualstma_loss(
                pred_pos, pred_vel, pred_heading,
                gt_pos, gt_vel, gt_heading
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Check for NaN
            if torch.isnan(loss):
                print(f'  WARNING: NaN loss at batch {n_batches}!')
                print(f'    pos_l={pos_l.item():.6f} vel_l={vel_l.item():.6f} head_l={head_l.item():.6f}')
                continue

            total_loss += loss.item()
            n_batches  += 1

            if train and n_batches % 100 == 0:
                print(f'  Epoch {epoch} | Batch {n_batches} | '
                      f'Loss {total_loss/n_batches:.6f} | '
                      f'Pos {pos_l.item():.6f} | '
                      f'Vel {vel_l.item():.6f} | '
                      f'Head {head_l.item():.6f}')

            del obs, gt_pos, gt_vel, gt_heading, pred_pos, pred_vel, pred_heading
            torch.cuda.empty_cache()

    avg = total_loss / max(1, n_batches)

    if train:
        metrics['train_loss'].append(avg)
        print(f'TRAIN Epoch {epoch}: loss={avg:.6f}')
        if avg < constant_metrics['min_train_loss']:
            constant_metrics['min_train_loss']  = avg
            constant_metrics['min_train_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')
        torch.save(model.state_dict(), checkpoint_dir + 'last.pth')
        with open(checkpoint_dir + 'last_epoch.txt', 'w') as f:
            f.write(str(epoch))
    else:
        metrics['val_loss'].append(avg)
        print(f'VALD  Epoch {epoch}: loss={avg:.6f}')
        if avg < constant_metrics['min_val_loss']:
            constant_metrics['min_val_loss']  = avg
            constant_metrics['min_val_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')

    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_set = os.path.join('./dataset', args.dataset)

    dset_train = DualSTMADataset(
        os.path.join(data_set, 'dualstma_train'),
        obs_len=args.obs_len, pred_len=args.pred_len,
        max_surr=args.max_surr
    )
    dset_val = DualSTMADataset(
        os.path.join(data_set, 'dualstma_val'),
        obs_len=args.obs_len, pred_len=args.pred_len,
        max_surr=args.max_surr
    )

    loader_train = DataLoader(
        dset_train, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    loader_val = DataLoader(
        dset_val, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    model = DualSTMA(
        d_model=32, hidden_dim=128, num_heads=8,
        num_layers=4, dropout=0.1,
        lstm_hidden=64, lstm_layers=2,
        pred_len=args.pred_len,
        num_vessel_types=101,
        num_lengths=32,
        num_widths=25,
        type_embed_dim=8
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)

    checkpoint_dir = f'./checkpoints/{args.tag}/{args.dataset}/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    if args.resume:
        last_ckpt       = checkpoint_dir + 'last.pth'
        last_epoch_file = checkpoint_dir + 'last_epoch.txt'
        if os.path.exists(last_ckpt):
            model.load_state_dict(torch.load(last_ckpt, map_location=device))
            with open(last_epoch_file) as f:
                start_epoch = int(f.read()) + 1
            print(f'Resumed from epoch {start_epoch}')

    with open(checkpoint_dir + 'args.pkl', 'wb') as f:
        pickle.dump(args, f)

    print(f'Training on {device}')
    print(f'Train: {len(dset_train):,} | Val: {len(dset_val):,}')

    for epoch in range(start_epoch, args.num_epochs):
        run_epoch(model, loader_train, optimizer, checkpoint_dir, epoch, train=True)
        run_epoch(model, loader_val,   optimizer, checkpoint_dir, epoch, train=False)

        print('*' * 40)
        print(f'Epoch {epoch}/{args.num_epochs}')
        for k, v in metrics.items():
            if v:
                print(f'  {k}: {v[-1]:.6f}')
        print(constant_metrics)
        print('*' * 40)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)

    print('Training complete!')


if __name__ == '__main__':
    log_path = './Logs_train/'
    os.makedirs(log_path, exist_ok=True)
    log_file = log_path + 'dualstma_v3-' + time.strftime('%Y%m%d-%H%M%S') + '.log'
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    main()