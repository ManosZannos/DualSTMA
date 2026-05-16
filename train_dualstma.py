"""
train_dualstma.py

Training script for DualSTMA (Huang et al., JMSE 2024).

Hyperparameters (Section 4.1.1):
  lr=0.0015, batch_size=256, epochs=100, Adam optimizer
  λ_position=10, λ_heading=1, λ_velocity=0.1

Usage (RTX 4090):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5

Usage (ihatz A100):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5 --gpu_num 0
"""

import os
import sys
import time
import argparse
import pickle
import glob

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
import pandas as pd

from model_dualstma import DualSTMA

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num',    default="",    type=str)
parser.add_argument('--dataset',    default='marinecadastre_2021')
parser.add_argument('--obs_len',    type=int, default=10)
parser.add_argument('--pred_len',   type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr',         type=float, default=0.0015)
parser.add_argument('--tag',        default='DualSTMA')
parser.add_argument('--resume',     action='store_true', default=False)
parser.add_argument('--max_surr',   type=int, default=10,
                    help='Max surrounding vessels per sample')

args_early, _ = parser.parse_known_args()
if args_early.gpu_num:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_early.gpu_num
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Training DualSTMA ...')
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
    Loads DualSTMA CSV and builds (obs, pred) sliding windows.

    Dynamic channels (Eq. 1):
      [lon, lat, dlon, dlat, v_lon, v_lat, θ_lon, θ_lat]
    """

    def __init__(self, data_dir, obs_len=10, pred_len=5, max_surr=10):
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.max_surr = max_surr
        self.seq_len  = obs_len + pred_len

        # Load all CSV files
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

        # Build index of (vessel_id, start_idx) for sliding windows
        self.samples = self._build_samples()
        print(f'  Total samples: {len(self.samples):,}')

    def _build_samples(self):
        samples = []
        vessel_groups = self.df.groupby('vessel_id')

        for vessel_id, group in vessel_groups:
            group = group.sort_values('frame_id').reset_index(drop=True)
            n = len(group)
            if n < self.seq_len:
                continue

            for start in range(n - self.seq_len + 1):
                end = start + self.seq_len
                frame_ids = group.iloc[start:end]['frame_id'].values
                diffs = np.diff(frame_ids)
                if not np.all(diffs == 1):
                    continue
                samples.append((vessel_id, start))

        return samples

    def _get_dynamic_features(self, rows):
        """
        Build 8-channel dynamic features from DataFrame rows.
        Channels: [lon, lat, dlon, dlat, v_lon, v_lat, θ_lon, θ_lat]
        """
        lon     = rows['LON'].values.astype(np.float32)
        lat     = rows['LAT'].values.astype(np.float32)
        sog     = rows['SOG'].values.astype(np.float32)
        heading = rows['Heading'].values.astype(np.float32)

        dlon = np.zeros_like(lon)
        dlat = np.zeros_like(lat)
        dlon[1:] = lon[1:] - lon[:-1]
        dlat[1:] = lat[1:] - lat[:-1]

        heading_rad = heading * 2 * np.pi
        v_lon     = sog * np.sin(heading_rad)
        v_lat     = sog * np.cos(heading_rad)
        theta_lon = np.sin(heading_rad)
        theta_lat = np.cos(heading_rad)

        return np.stack([lon, lat, dlon, dlat, v_lon, v_lat, theta_lon, theta_lat], axis=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vessel_id, start = self.samples[idx]

        # Get vessel rows
        vessel_rows = self.df[self.df['vessel_id'] == vessel_id].sort_values('frame_id').reset_index(drop=True)
        window = vessel_rows.iloc[start:start + self.seq_len]

        obs_rows  = window.iloc[:self.obs_len]
        pred_rows = window.iloc[self.obs_len:self.obs_len + self.pred_len]

        # Verify pred_len
        assert len(pred_rows) == self.pred_len, f"Expected {self.pred_len} pred rows, got {len(pred_rows)}"

        # Dynamic features
        all_features  = self._get_dynamic_features(window)         # [seq_len, 8]
        obs_features  = all_features[:self.obs_len]                 # [obs_len, 8]

        # Static features
        v_type   = int(obs_rows['vessel_type'].iloc[0])
        v_length = int(obs_rows['vessel_length'].iloc[0])
        v_width  = int(obs_rows['vessel_width'].iloc[0])

        # Ground truth
        gt_lon = pred_rows['LON'].values.astype(np.float32)
        gt_lat = pred_rows['LAT'].values.astype(np.float32)
        gt_pos = np.stack([gt_lon, gt_lat], axis=-1)               # [pred_len, 2]

        pred_features = all_features[self.obs_len:self.obs_len + self.pred_len]
        gt_vel        = pred_features[:, 2:4]                       # dlon, dlat
        gt_heading    = pred_features[:, 6:8]                       # θ_lon, θ_lat

        # Last observed position (for vessel-centered translation)
        last_obs_pos = obs_features[-1, :2].copy()                  # [2]

        # Vessel-centered: translate so last obs = origin
        obs_centered = obs_features.copy()
        obs_centered[:, 0] -= last_obs_pos[0]
        obs_centered[:, 1] -= last_obs_pos[1]

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
        surr_lengths = []
        surr_widths  = []

        for sv in surr_vessels[:self.max_surr]:
            sv_rows = surr_df[surr_df['vessel_id'] == sv].sort_values('frame_id')
            sv_window = sv_rows[sv_rows['frame_id'].isin(obs_frame_ids)].sort_values('frame_id')
            if len(sv_window) != self.obs_len:
                continue
            sv_feat = self._get_dynamic_features(sv_window)         # [obs_len, 8]
            sv_feat[:, 0] -= last_obs_pos[0]
            sv_feat[:, 1] -= last_obs_pos[1]
            surr_features_list.append(sv_feat)
            surr_types.append(int(sv_window['vessel_type'].iloc[0]))
            surr_lengths.append(int(sv_window['vessel_length'].iloc[0]))
            surr_widths.append(int(sv_window['vessel_width'].iloc[0]))

        # Pad to max_surr
        surr_dynamic = np.zeros((self.max_surr, self.obs_len, 8), dtype=np.float32)
        s_types      = np.zeros(self.max_surr, dtype=np.int64)
        s_lengths    = np.zeros(self.max_surr, dtype=np.int64)
        s_widths     = np.zeros(self.max_surr, dtype=np.int64)

        for i, feat in enumerate(surr_features_list):
            surr_dynamic[i] = feat
            s_types[i]      = surr_types[i]
            s_lengths[i]    = surr_lengths[i]
            s_widths[i]     = surr_widths[i]

        return {
            'obs_features':  torch.tensor(obs_centered,  dtype=torch.float32),  # [obs_len, 8]
            'gt_pos':        torch.tensor(gt_pos,         dtype=torch.float32),  # [pred_len, 2]
            'gt_vel':        torch.tensor(gt_vel,         dtype=torch.float32),  # [pred_len, 2]
            'gt_heading':    torch.tensor(gt_heading,     dtype=torch.float32),  # [pred_len, 2]
            'last_obs_pos':  torch.tensor(last_obs_pos,   dtype=torch.float32),  # [2]
            'v_type':        torch.tensor(v_type,         dtype=torch.long),
            'v_length':      torch.tensor(v_length,       dtype=torch.long),
            'v_width':       torch.tensor(v_width,        dtype=torch.long),
            'surr_dynamic':  torch.tensor(surr_dynamic,   dtype=torch.float32),  # [max_surr, obs_len, 8]
            's_types':       torch.tensor(s_types,        dtype=torch.long),
            's_lengths':     torch.tensor(s_lengths,      dtype=torch.long),
            's_widths':      torch.tensor(s_widths,       dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def wrapped_heading_error(pred_h, gt_h):
    """Wrap heading error to [-π, π] (Eq. 38)."""
    diff = pred_h - gt_h
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def dualstma_loss(pred_pos, pred_vel, pred_heading,
                  gt_pos, gt_vel, gt_heading,
                  last_obs_pos,
                  lambda_pos=10.0, lambda_heading=1.0, lambda_vel=0.1):
    """
    DualSTMA loss (Section 3.4, Eq. 36-41):
      L = λ_pos * L_position + λ_heading * L_heading + λ_vel * L_velocity
    """
    pred_len = pred_pos.shape[1]

    # Position loss — short-term weighted (Eq. 36)
    abs_pred = pred_pos + last_obs_pos.unsqueeze(1)
    abs_gt   = gt_pos

    weights = torch.tensor(
        [(pred_len - t) for t in range(pred_len)],
        dtype=torch.float32, device=pred_pos.device
    )
    weights = weights / weights.sum()

    pos_loss = sum(
        weights[t] * F.huber_loss(abs_pred[:, t], abs_gt[:, t])
        for t in range(pred_len)
    )

    # Velocity loss (Eq. 40)
    vel_loss = F.huber_loss(pred_vel, gt_vel)

    # Heading loss (Eq. 39)
    h_error      = wrapped_heading_error(pred_heading, gt_heading)
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


def run_epoch(model, loader, optimizer, scaler, checkpoint_dir, epoch, train=True):
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

            v_type   = batch['v_type'].to(device)
            v_length = batch['v_length'].to(device)
            v_width  = batch['v_width'].to(device)

            surr_dyn  = batch['surr_dynamic'].to(device)
            s_types   = batch['s_types'].to(device)
            s_lengths = batch['s_lengths'].to(device)
            s_widths  = batch['s_widths'].to(device)

            target_static      = (v_type, v_length, v_width)
            surrounding_static = (s_types, s_lengths, s_widths)

            with autocast():
                pred_pos, pred_vel, pred_heading = model(
                    obs, target_static,
                    surr_dyn, surrounding_static
                )

                loss, pos_l, vel_l, head_l = dualstma_loss(
                    pred_pos, pred_vel, pred_heading,
                    gt_pos, gt_vel, gt_heading, last_obs
                )

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

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
        shuffle=True, num_workers=4, pin_memory=True
    )
    loader_val = DataLoader(
        dset_val, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler    = GradScaler()

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
    print(f'Train: {len(dset_train)} samples | Val: {len(dset_val)} samples')

    for epoch in range(start_epoch, args.num_epochs):
        run_epoch(model, loader_train, optimizer, scaler, checkpoint_dir, epoch, train=True)
        run_epoch(model, loader_val,   optimizer, scaler, checkpoint_dir, epoch, train=False)

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
    log_file = log_path + 'dualstma-' + time.strftime('%Y%m%d-%H%M%S') + '.log'
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    main()