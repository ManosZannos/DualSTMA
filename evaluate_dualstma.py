"""
evaluate_dualstma.py

Evaluation script for DualSTMA (Huang et al., JMSE 2024).

Metrics (Section 4.1.4, Eq. 42-45) in degrees:
  RMSE, MAE, ADE, FDE

Usage (ihatz):
  python evaluate_dualstma.py \
    --dataset marinecadastre_2021 \
    --checkpoint /storage/data4/ihatz/dualstma/checkpoints/DualSTMA_v3/marinecadastre_2021/val_best.pth \
    --split test
"""

import os
import json
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dualstma import DualSTMA
from train_dualstma import DualSTMADataset

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num',    default='',   type=str)
parser.add_argument('--dataset',    default='marinecadastre_2021')
parser.add_argument('--checkpoint', required=True, type=str)
parser.add_argument('--split',      default='test', choices=['test', 'val'])
parser.add_argument('--obs_len',    type=int, default=10)
parser.add_argument('--pred_len',   type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_surr',   type=int, default=10)

args = parser.parse_args()

if args.gpu_num:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ---------------------------------------------------------------------------
# Denormalization
# ---------------------------------------------------------------------------

def load_global_stats(dataset):
    stats_path = os.path.join('./dataset', dataset, 'global_stats.json')
    with open(stats_path) as f:
        stats = json.load(f)
    lon_min   = stats['LON']['mean']
    lon_range = stats['LON']['std']
    lat_min   = stats['LAT']['mean']
    lat_range = stats['LAT']['std']
    return lon_min, lon_range, lat_min, lat_range


def denormalize(lon_norm, lat_norm, lon_min, lon_range, lat_min, lat_range):
    lon = lon_norm * lon_range + lon_min
    lat = lat_norm * lat_range + lat_min
    return lon, lat


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, loader):
    """Predictions and gt_pos are in degrees (LON_abs/LAT_abs) — no denormalization needed."""
    model.eval()

    all_pred_lon = []
    all_pred_lat = []
    all_gt_lon   = []
    all_gt_lat   = []

    with torch.no_grad():
        for batch in loader:
            obs        = batch['obs_features'].to(device)
            gt_pos     = batch['gt_pos'].to(device)
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

            pred_pos, _, _ = model(
                obs, target_static,
                head_rad, last_obs,
                surr_dyn, surrounding_static,
                surr_mask=surr_mask
            )

            # Already in degrees (LON_abs/LAT_abs)
            pred_lon = pred_pos[:, :, 0].cpu().numpy()
            pred_lat = pred_pos[:, :, 1].cpu().numpy()
            gt_lon   = gt_pos[:, :, 0].cpu().numpy()
            gt_lat   = gt_pos[:, :, 1].cpu().numpy()

            all_pred_lon.append(pred_lon)
            all_pred_lat.append(pred_lat)
            all_gt_lon.append(gt_lon)
            all_gt_lat.append(gt_lat)

    pred_lon = np.concatenate(all_pred_lon, axis=0)
    pred_lat = np.concatenate(all_pred_lat, axis=0)
    gt_lon   = np.concatenate(all_gt_lon,   axis=0)
    gt_lat   = np.concatenate(all_gt_lat,   axis=0)

    N, T = pred_lon.shape
    dlon = pred_lon - gt_lon
    dlat = pred_lat - gt_lat

    # Metrics (Eq. 42-45)
    rmse = np.sqrt(np.mean(dlon**2 + dlat**2))
    mae  = np.mean(np.abs(dlon) + np.abs(dlat))
    ade  = np.mean(np.sqrt(dlon**2 + dlat**2))
    fde  = np.mean(np.sqrt(dlon[:, -1]**2 + dlat[:, -1]**2))

    ade_per_step = [
        np.mean(np.sqrt(dlon[:, t]**2 + dlat[:, t]**2))
        for t in range(T)
    ]

    return rmse, mae, ade, fde, ade_per_step, N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_set = os.path.join('./dataset', args.dataset)

    print('Using LON_abs/LAT_abs (degrees) — no denormalization needed')

    split_dir = os.path.join(data_set, f'dualstma_{args.split}')
    dset = DualSTMADataset(
        split_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_surr=args.max_surr
    )
    loader = DataLoader(
        dset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    model = DualSTMA(
        d_model=32, hidden_dim=128, num_heads=8,
        num_layers=4, dropout=0.1,
        lstm_hidden=64, lstm_layers=2,
        pred_len=args.pred_len,
        num_vessel_types=101, num_lengths=32, num_widths=25,
        type_embed_dim=8
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print(f'Loaded: {args.checkpoint}')

    rmse, mae, ade, fde, ade_per_step, N = evaluate(model, loader)

    horizons = [(t+1)*10 for t in range(args.pred_len)]

    print('\n' + '='*70)
    print('EVALUATION RESULTS — DualSTMA v3')
    print('='*70)
    print(f'Dataset:   {args.dataset} ({args.split})')
    print(f'Sequences: {N:,}')
    print(f'Checkpoint: {os.path.basename(args.checkpoint)}')
    print()
    print(f'{"Horizon":>10} | {"ADE (°)":>12}')
    print('-'*30)
    for h, a in zip(horizons, ade_per_step):
        print(f'  ADE {h:2d}min | {a:12.6f}°')
    print('-'*30)
    print(f'{"RMSE":>10} | {rmse:12.6f}°')
    print(f'{"MAE":>10} | {mae:12.6f}°')
    print(f'{"ADE":>10} | {ade:12.6f}°')
    print(f'{"FDE":>10} | {fde:12.6f}°')
    print('='*70)
    print('\nPaper reference (M5):')
    print(f'  RMSE: 0.004223°  MAE: 0.003021°  ADE: 0.002436°  FDE: 0.003946°')

    out_dir  = os.path.dirname(args.checkpoint)
    out_file = os.path.join(out_dir, f'eval_{args.split}.txt')
    with open(out_file, 'w') as f:
        f.write(f'Dataset: {args.dataset} ({args.split})\n')
        f.write(f'Sequences: {N}\n')
        f.write(f'Checkpoint: {args.checkpoint}\n\n')
        for h, a in zip(horizons, ade_per_step):
            f.write(f'ADE {h}min: {a:.6f}°\n')
        f.write(f'\nRMSE: {rmse:.6f}°\n')
        f.write(f'MAE:  {mae:.6f}°\n')
        f.write(f'ADE:  {ade:.6f}°\n')
        f.write(f'FDE:  {fde:.6f}°\n')
    print(f'\nSaved: {out_file}')


if __name__ == '__main__':
    main()