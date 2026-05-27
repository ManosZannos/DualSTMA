"""
train_dualstma.py

Training script for DualSTMA — Simplified (Επιλογή Α)
Huang et al., JMSE 2024

ΑΛΛΑΓΕΣ από v4:
1. Dataset: 4 features [LON, LAT, SOG, Heading] αντί 8 vessel-centered
   - Δεν γίνεται rotation/translation
   - gt_pos: absolute normalized [0,1] positions (όπως METO-S2S)
   - Αφαίρεση gt_vel, gt_heading, heading_rad, last_obs_pos
2. Loss: μόνο position Huber (χωρίς vel/heading terms)
3. Model forward: χωρίς heading_rad / last_obs_pos arguments

Usage (RTX 4090):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5 --tag DualSTMA_simplified

Usage (ihatz A100):
  python train_dualstma.py --dataset marinecadastre_2021 --obs_len 10 --pred_len 5 --gpu_num 0 --tag DualSTMA_simplified
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
from torch.nn import functional as F
import pandas as pd

from model_dualstma import DualSTMA

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
parser.add_argument('--tag',        default='DualSTMA_simplified')
parser.add_argument('--resume',     action='store_true', default=False)
parser.add_argument('--max_surr',   type=int, default=10)

args_early, _ = parser.parse_known_args()
if args_early.gpu_num:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_early.gpu_num
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Training DualSTMA Simplified (4 features, absolute positions, position-only loss) ...')
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
    DualSTMA dataset — Simplified (Επιλογή Α)

    ΑΛΛΑΓΕΣ:
    - 4 features: [LON, LAT, SOG, Heading] normalized [0,1]
    - Χωρίς vessel-centered rotation/translation
    - gt_pos: absolute normalized [0,1] positions
    - Χωρίς gt_vel, gt_heading, heading_rad, last_obs_pos
    - surr_mask: True=real vessel, False=padded (διατηρείται)
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

    def _get_dynamic_features(self, rows):
        """
        4 features: [LON, LAT, SOG, Heading] — normalized [0,1]
        Όπως METO-S2S — χωρίς vessel-centered transform.
        """
        lon     = rows['LON'].values.astype(np.float32)
        lat     = rows['LAT'].values.astype(np.float32)
        sog     = rows['SOG'].values.astype(np.float32)
        heading = rows['Heading'].values.astype(np.float32)
        return np.stack([lon, lat, sog, heading], axis=-1)  # [T, 4]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vessel_id, start = self.samples[idx]

        vessel_rows = (self.df[self.df['vessel_id'] == vessel_id]
                       .sort_values('frame_id')
                       .reset_index(drop=True))
        window   = vessel_rows.iloc[start:start + self.seq_len]
        obs_rows = window.iloc[:self.obs_len]
        pred_rows = window.iloc[self.obs_len:self.obs_len + self.pred_len]

        assert len(pred_rows) == self.pred_len

        # --- Target vessel dynamic features (observation window) ---
        obs_features = self._get_dynamic_features(obs_rows)  # [obs_len, 4]

        # --- Static features ---
        v_type   = int(obs_rows['vessel_type'].iloc[0])
        v_width  = int(obs_rows['vessel_width'].iloc[0])
        v_length = int(obs_rows['vessel_length'].iloc[0])

        # --- Ground truth: absolute normalized positions [0,1] ---
        gt_lon = pred_rows['LON'].values.astype(np.float32)
        gt_lat = pred_rows['LAT'].values.astype(np.float32)
        gt_pos = np.stack([gt_lon, gt_lat], axis=-1)  # [pred_len, 2]

        # --- Surrounding vessels ---
        obs_frame_ids   = obs_rows['frame_id'].values
        frame_start_obs = obs_frame_ids[0]
        frame_end_obs   = obs_frame_ids[-1]

        surr_df = self.df[
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
            # Surrounding vessels: ίδια 4 features, χωρίς rotation
            sv_feat = self._get_dynamic_features(sv_window)  # [obs_len, 4]
            surr_features_list.append(sv_feat)
            surr_types.append(int(sv_window['vessel_type'].iloc[0]))
            surr_widths.append(int(sv_window['vessel_width'].iloc[0]))
            surr_lengths.append(int(sv_window['vessel_length'].iloc[0]))

        # Padding
        surr_dynamic = np.zeros((self.max_surr, self.obs_len, 4), dtype=np.float32)
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
            'obs_features': torch.tensor(obs_features,  dtype=torch.float32),
            'gt_pos':       torch.tensor(gt_pos,         dtype=torch.float32),
            'v_type':       torch.tensor(v_type,   dtype=torch.long),
            'v_width':      torch.tensor(v_width,  dtype=torch.long),
            'v_length':     torch.tensor(v_length, dtype=torch.long),
            'surr_dynamic': torch.tensor(surr_dynamic, dtype=torch.float32),
            's_types':      torch.tensor(s_types,   dtype=torch.long),
            's_widths':     torch.tensor(s_widths,  dtype=torch.long),
            's_lengths':    torch.tensor(s_lengths, dtype=torch.long),
            'surr_mask':    torch.tensor(surr_mask, dtype=torch.bool),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dualstma_loss(pred_pos, gt_pos, lambda_pos=10.0):
    """
    Position-only Huber loss με time-decay weights.
    ΑΛΛΑΓΗ: αφαίρεση vel_loss και heading_loss.

    pred_pos: [N, pred_len, 2] — normalized [0,1]
    gt_pos:   [N, pred_len, 2] — normalized [0,1]
    """
    pred_len = pred_pos.shape[1]

    # Time-decay weights: πρώτα timesteps πιο σημαντικά
    weights = torch.tensor(
        [(pred_len - t) for t in range(pred_len)],
        dtype=torch.float32, device=pred_pos.device
    )
    weights = weights / weights.sum()

    pos_loss = sum(
        weights[t] * F.huber_loss(pred_pos[:, t], gt_pos[:, t])
        for t in range(pred_len)
    )

    total = lambda_pos * pos_loss
    return total, pos_loss


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

    total_loss     = 0.0
    total_pos_loss = 0.0
    n_batches      = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            obs        = batch['obs_features'].to(device)   # [N, obs_len, 4]
            gt_pos     = batch['gt_pos'].to(device)          # [N, pred_len, 2]

            v_type   = batch['v_type'].to(device)
            v_width  = batch['v_width'].to(device)
            v_length = batch['v_length'].to(device)

            surr_dyn  = batch['surr_dynamic'].to(device)    # [N, max_surr, obs_len, 4]
            s_types   = batch['s_types'].to(device)
            s_widths  = batch['s_widths'].to(device)
            s_lengths = batch['s_lengths'].to(device)
            surr_mask = batch['surr_mask'].to(device)

            target_static      = (v_type, v_width, v_length)
            surrounding_static = (s_types, s_widths, s_lengths)

            # Forward — χωρίς heading_rad / last_obs_pos
            pred_pos = model(
                obs, target_static,
                surr_dyn, surrounding_static,
                surr_mask=surr_mask
            )

            loss, pos_l = dualstma_loss(pred_pos, gt_pos)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if torch.isnan(loss):
                print(f'  WARNING: NaN loss at batch {n_batches}!')
                continue

            total_loss     += loss.item()
            total_pos_loss += pos_l.item()
            n_batches      += 1

            if train and n_batches % 100 == 0:
                print(f'  Epoch {epoch} | Batch {n_batches} | '
                      f'Loss {total_loss/n_batches:.6f} | '
                      f'Pos {total_pos_loss/n_batches:.6f}')

            del obs, gt_pos, pred_pos
            torch.cuda.empty_cache()

    avg     = total_loss / max(1, n_batches)
    avg_pos = total_pos_loss / max(1, n_batches)

    if train:
        metrics['train_loss'].append(avg)
        print(f'TRAIN Epoch {epoch}: loss={avg:.6f} | pos_loss={avg_pos:.6f}')
        if avg < constant_metrics['min_train_loss']:
            constant_metrics['min_train_loss']  = avg
            constant_metrics['min_train_epoch'] = epoch
            torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')
        torch.save(model.state_dict(), checkpoint_dir + 'last.pth')
        with open(checkpoint_dir + 'last_epoch.txt', 'w') as f:
            f.write(str(epoch))
    else:
        metrics['val_loss'].append(avg)
        print(f'VALD  Epoch {epoch}: loss={avg:.6f} | pos_loss={avg_pos:.6f}')
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
    log_file = log_path + 'dualstma_simplified-' + time.strftime('%Y%m%d-%H%M%S') + '.log'
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    main()