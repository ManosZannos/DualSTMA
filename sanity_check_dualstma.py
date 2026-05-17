"""
sanity_check_dualstma.py

Sanity checks for DualSTMA v2 before training:
  1. Shape check — forward pass produces correct output shapes
  2. Inverse transform check — rotate → inverse rotate = identity
  3. Loss decrease check — loss decreases over a few steps on dummy data

Usage:
  python sanity_check_dualstma.py
"""

import math
import torch
import torch.nn as nn
import numpy as np
from model_dualstma import DualSTMA, rotate_translate, inverse_rotate_translate, rotate_vector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('=' * 60)

# ---------------------------------------------------------------------------
# Test 1: Shape check
# ---------------------------------------------------------------------------
print('\n[TEST 1] Shape check')

N        = 4    # batch size
T        = 10   # obs_len
M        = 3    # surrounding vessels
pred_len = 5

model = DualSTMA(
    d_model=32, hidden_dim=128, num_heads=8,
    num_layers=4, dropout=0.1,
    lstm_hidden=64, lstm_layers=2,
    pred_len=pred_len,
    num_vessel_types=101,
    num_lengths=32,
    num_widths=25,
    type_embed_dim=8
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Model parameters: {n_params:,}')

# Dummy inputs
target_dynamic = torch.randn(N, T, 8).to(device)
heading_rad    = torch.rand(N).to(device) * 2 * math.pi
last_obs_pos   = torch.randn(N, 2).to(device)

v_type   = torch.randint(0, 100, (N,)).to(device)
v_width  = torch.randint(0, 24,  (N,)).to(device)
v_length = torch.randint(1, 31,  (N,)).to(device)
target_static = (v_type, v_width, v_length)

surr_dynamic = torch.randn(N, M, T, 8).to(device)
s_types   = torch.randint(0, 100, (N, M)).to(device)
s_widths  = torch.randint(0, 24,  (N, M)).to(device)
s_lengths = torch.randint(1, 31,  (N, M)).to(device)
surrounding_static = (s_types, s_widths, s_lengths)

with torch.no_grad():
    pos, vel, heading = model(
        target_dynamic, target_static,
        heading_rad, last_obs_pos,
        surr_dynamic, surrounding_static
    )

expected_shape = (N, pred_len, 2)
assert pos.shape     == expected_shape, f'pos shape: {pos.shape} != {expected_shape}'
assert vel.shape     == expected_shape, f'vel shape: {vel.shape} != {expected_shape}'
assert heading.shape == expected_shape, f'heading shape: {heading.shape} != {expected_shape}'

print(f'  pos shape:     {pos.shape} ✓')
print(f'  vel shape:     {vel.shape} ✓')
print(f'  heading shape: {heading.shape} ✓')
print('  [TEST 1] PASSED ✓')

# ---------------------------------------------------------------------------
# Test 2: Inverse transform check
# ---------------------------------------------------------------------------
print('\n[TEST 2] Inverse transform check (rotate → inverse rotate = identity)')

# Create test coordinates
coords = torch.tensor([[0.5, 0.3], [-0.2, 0.7], [0.1, -0.4]], dtype=torch.float32)
origin = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype=torch.float32)
heading_test = torch.tensor([math.pi / 4, math.pi / 4, math.pi / 4])  # 45 degrees

# Apply forward transform
coords_transformed = rotate_translate(coords, heading_test, origin)

# Apply inverse transform
coords_recovered = inverse_rotate_translate(
    coords_transformed,
    heading_test,
    origin
)

max_error = (coords_recovered - coords).abs().max().item()
print(f'  Max reconstruction error: {max_error:.2e}')
assert max_error < 1e-5, f'Inverse transform error too large: {max_error}'
print('  [TEST 2] PASSED ✓')

# Test with batch dimension [N, pred_len, 2]
N_test    = 8
pred_test = 5
coords_batch  = torch.randn(N_test, pred_test, 2)
origin_batch  = torch.randn(N_test, 2)
heading_batch = torch.rand(N_test) * 2 * math.pi

# Expand origin and heading for batch
origin_expanded  = origin_batch.unsqueeze(1).expand(-1, pred_test, -1)
heading_expanded = heading_batch.unsqueeze(1).expand(-1, pred_test)

coords_t = rotate_translate(coords_batch, heading_expanded, origin_expanded)
coords_r = inverse_rotate_translate(coords_t, heading_expanded, origin_expanded)

max_error_batch = (coords_r - coords_batch).abs().max().item()
print(f'  Max batch reconstruction error: {max_error_batch:.2e}')
assert max_error_batch < 1e-5, f'Batch inverse transform error: {max_error_batch}'
print('  Batch inverse transform [TEST 2] PASSED ✓')

# ---------------------------------------------------------------------------
# Test 3: Loss decrease check
# ---------------------------------------------------------------------------
print('\n[TEST 3] Loss decrease check (10 steps on dummy data)')

from torch.nn import functional as F

def wrapped_heading_error_floormod(pred_h, gt_h):
    diff = pred_h - gt_h
    return -math.pi + torch.fmod(diff + math.pi, 2 * math.pi)

def dualstma_loss(pred_pos, pred_vel, pred_heading,
                  gt_pos, gt_vel, gt_heading,
                  lambda_pos=10.0, lambda_heading=1.0, lambda_vel=0.1):
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
    return total

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# Fixed dummy targets
gt_pos     = torch.randn(N, pred_len, 2).to(device)
gt_vel     = torch.randn(N, pred_len, 2).to(device)
gt_heading = torch.randn(N, pred_len, 2).to(device)

losses = []
for step in range(10):
    optimizer.zero_grad()
    pred_pos, pred_vel, pred_heading = model(
        target_dynamic, target_static,
        heading_rad, last_obs_pos,
        surr_dynamic, surrounding_static
    )
    loss = dualstma_loss(pred_pos, pred_vel, pred_heading,
                         gt_pos, gt_vel, gt_heading)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    print(f'  Step {step+1:2d}: loss = {loss.item():.6f}')

# Check that loss decreased overall
assert losses[-1] < losses[0], \
    f'Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}'
print(f'\n  Loss decreased: {losses[0]:.6f} → {losses[-1]:.6f} ✓')
print('  [TEST 3] PASSED ✓')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n' + '=' * 60)
print('ALL TESTS PASSED ✓')
print('Model is ready for training.')
print('=' * 60)
