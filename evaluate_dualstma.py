"""
evaluate_dualstma.py

Evaluation script για DualSTMA Simplified.
Υπολογίζει ADE/FDE σε degrees (denormalized) για σύγκριση με paper.

Denormalization constants (από global_stats.json / METO-S2S):
  LON: actual = norm * 72.60811 + (-133.29703)
  LAT: actual = norm * 28.32044 + 20.90883

Usage:
  python evaluate_dualstma.py \
    --dataset marinecadastre_2021 \
    --checkpoint checkpoints/DualSTMA_simplified/marinecadastre_2021/val_best.pth \
    --split test
"""

import os
import argparse
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader

from model_dualstma import DualSTMA
from train_dualstma import DualSTMADataset

# ---------------------------------------------------------------------------
# Denormalization constants (METO-S2S global_stats.json)
# ---------------------------------------------------------------------------
LON_MIN   = -133.29703
LON_RANGE =   72.60811
LAT_MIN   =   20.90883
LAT_RANGE =   28.32044


def denorm_lon(x):
    return x * LON_RANGE + LON_MIN


def denorm_lat(x):
    return x * LAT_RANGE + LAT_MIN


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num',    default="",   type=str)
parser.add_argument('--dataset',    default='marinecadastre_2021')
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--split',      default='test', choices=['val', 'test'])
parser.add_argument('--obs_len',    type=int, default=10)
parser.add_argument('--pred_len',   type=int, default=5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_surr',   type=int, default=10)

args = parser.parse_args()

if args.gpu_num:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate():
    data_dir = os.path.join('./dataset', args.dataset, f'dualstma_{args.split}')

    dataset = DualSTMADataset(
        data_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        max_surr=args.max_surr
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
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

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'Evaluating on {args.split} set ({len(dataset):,} samples) ...')

    # ADE per horizon: [pred_len]
    ade_per_horizon = np.zeros(args.pred_len)
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            obs        = batch['obs_features'].to(device)
            gt_pos     = batch['gt_pos'].to(device)         # [N, pred_len, 2] normalized

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

            pred_pos = model(
                obs, target_static,
                surr_dyn, surrounding_static,
                surr_mask=surr_mask
            )  # [N, pred_len, 2] normalized

            # Denormalize → degrees
            pred_lon = denorm_lon(pred_pos[:, :, 0].cpu().numpy())
            pred_lat = denorm_lat(pred_pos[:, :, 1].cpu().numpy())
            gt_lon   = denorm_lon(gt_pos[:, :, 0].cpu().numpy())
            gt_lat   = denorm_lat(gt_pos[:, :, 1].cpu().numpy())

            # Euclidean distance in degrees per horizon
            dist = np.sqrt((pred_lon - gt_lon) ** 2 + (pred_lat - gt_lat) ** 2)
            # dist: [N, pred_len]

            ade_per_horizon += dist.sum(axis=0)
            n_samples += dist.shape[0]

    ade_per_horizon /= n_samples
    ade_avg = ade_per_horizon.mean()
    fde     = ade_per_horizon[-1]

    print('\n' + '=' * 50)
    print(f'Results on {args.split} set:')
    print('=' * 50)
    for t, ade_t in enumerate(ade_per_horizon):
        minutes = (t + 1) * 10
        print(f'  ADE {minutes:3d}min: {ade_t:.6f}°')
    print(f'  ADE (avg):  {ade_avg:.6f}°')
    print(f'  FDE:        {fde:.6f}°')
    print('=' * 50)
    print(f'\nDualSTMA paper: ADE=0.002436°, FDE=0.003946°')
    print(f'Gap vs paper:   ADE={ade_avg/0.002436:.1f}x, FDE={fde/0.003946:.1f}x')


if __name__ == '__main__':
    evaluate()