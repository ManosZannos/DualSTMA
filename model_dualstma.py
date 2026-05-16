"""
model_dualstma.py

DualSTMA: Dual Spatial-Temporal Multi-head Attention
Huang et al., Journal of Marine Science and Engineering, 2024
DOI: 10.3390/jmse12112031

Architecture:
  1. Feature Preprocessing
     - Dynamic: Conv2d + affine + ReLU (Eq. 1-3)
     - Static:  Embedding + MLP + Sigmoid gating (Eq. 4-5)
  2. Dual Encoder
     - TS path: temporal self-attention → spatial cross-attention (Eq. 6-14)
     - ST path: spatial cross-attention → temporal self-attention (Eq. 15-23)
  3. LSTM Decoder (Eq. 24-33)
  4. Loss: position + velocity + heading Huber loss (Eq. 36-41)

Hyperparameters (from Section 4.1.1):
  input_dim=32, hidden_dim=128, num_heads=8, num_layers=4
  dropout=0.1, lstm_hidden=64, lstm_layers=2
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x: [B, T, D]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Feature Preprocessing
# ---------------------------------------------------------------------------

class DynamicFeatureEncoder(nn.Module):
    """
    Encodes dynamic features per time step.
    Input: 8 channels [lon, lat, dlon, dlat, v_lon, v_lat, θ_lon, θ_lat]
    Output: [T, d_model] per vessel
    Paper Eq. 1-3: Conv2d + affine (γ,β) + ReLU
    """

    def __init__(self, in_channels=8, d_model=32):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.relu  = nn.ReLU()

    def forward(self, x):
        """
        x: [N, T, 8]
        Returns: [N, T, d_model]
        """
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [N, 8, T, 1]
        x = self.conv(x)                        # [N, d_model, T, 1]
        x = x.squeeze(-1).permute(0, 2, 1)      # [N, T, d_model]
        x = x * self.gamma + self.beta
        x = self.relu(x)
        return x


class StaticFeatureEncoder(nn.Module):
    """
    Encodes static vessel features → gating signal.
    Input: vessel_type (categorical), length (binned), width (binned)
    Output: gating signal in [0,1] — [N, d_model]
    Paper Eq. 4-5: Embedding + MLP + Sigmoid

    Embedding sizes verified from dataset:
      num_vessel_types=101  (max vessel_type=100)
      num_lengths=32        (max vessel_length=31)
      num_widths=25         (max vessel_width=24)
    """

    def __init__(self,
                 num_vessel_types=101,
                 num_lengths=32,
                 num_widths=25,
                 type_embed_dim=8,
                 d_model=32):
        super().__init__()

        self.type_embedding = nn.Embedding(num_vessel_types, type_embed_dim)

        mlp_in = type_embed_dim + 2  # type_emb + length + width
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, vessel_type, vessel_length, vessel_width):
        """
        vessel_type:   [N] int
        vessel_length: [N] int
        vessel_width:  [N] int
        Returns: gating [N, d_model], type_emb [N, type_embed_dim]
        """
        type_emb = self.type_embedding(vessel_type)    # [N, type_embed_dim]
        length   = vessel_length.float().unsqueeze(-1)  # [N, 1]
        width    = vessel_width.float().unsqueeze(-1)   # [N, 1]

        z = torch.cat([type_emb, length, width], dim=-1)
        z = self.mlp(z)
        z = self.sigmoid(z)
        return z, type_emb


# ---------------------------------------------------------------------------
# Transformer Building Blocks
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single Transformer layer with pre-norm, residual, dropout.
    Used for both self-attention and cross-attention.
    """

    def __init__(self, d_model=128, num_heads=8, dropout=0.1, cross_attention=False):
        super().__init__()
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, key_padding_mask=None):
        """
        x:       [B, T, d_model]
        context: [B, S, d_model] — for cross-attention
        """
        if self.cross_attention and context is not None:
            x2, _ = self.attn(
                self.norm1(x), context, context,
                key_padding_mask=key_padding_mask
            )
        else:
            x2, _ = self.attn(
                self.norm1(x), self.norm1(x), self.norm1(x),
                key_padding_mask=key_padding_mask
            )
        x = x + self.dropout(x2)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# TS Path: Temporal → Spatial
# ---------------------------------------------------------------------------

class TemporalSpatialPath(nn.Module):
    """
    Temporal-Spatial Path (Section 3.2.1):
    1. Self-attention over time series (with learnable aggregation token)
    2. Cross-attention over surrounding vessels (spatial)
    Paper Eq. 6-14
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.d_model   = d_model
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        self.temporal_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=False)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=True)
            for _ in range(num_layers)
        ])

    def forward(self, zd, zs_gate, type_emb, surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        zd:                  [N, T, D]
        zs_gate:             [N, D]
        surrounding_zd:      [N, M, T, D] or None
        surrounding_zs_gate: [N, M, D] or None
        Returns: TS [N, D]
        """
        N, T, D = zd.shape

        # Step 1: Temporal self-attention with aggregation token (Eq. 6-9)
        agg    = self.agg_token.expand(N, -1, -1)
        zd_hat = torch.cat([zd, agg], dim=1)       # [N, T+1, D]
        zd_hat = self.pos_enc(zd_hat)

        hd = zd_hat
        for layer in self.temporal_layers:
            hd = layer(hd)

        hd_i = hd[:, -1, :]   # [N, D] — aggregation token (Eq. 9)

        # Step 2: Gating (Eq. 10-11)
        hi = hd_i * zs_gate   # [N, D]

        # Step 3: Spatial cross-attention (Eq. 12-14)
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        if M > 0:
            surr_summaries = []
            for m in range(M):
                surr_zd_m  = surrounding_zd[:, m, :, :]      # [N, T, D]
                surr_agg   = self.agg_token.expand(N, -1, -1)
                surr_hat   = torch.cat([surr_zd_m, surr_agg], dim=1)
                surr_hat   = self.pos_enc(surr_hat)
                surr_h     = surr_hat
                for layer in self.temporal_layers:
                    surr_h = layer(surr_h)
                surr_summary = surr_h[:, -1, :]               # [N, D]
                surr_gate    = surrounding_zs_gate[:, m, :]   # [N, D]
                surr_summary = surr_summary * surr_gate
                surr_summaries.append(surr_summary.unsqueeze(1))

            surr_context = torch.cat(surr_summaries, dim=1)   # [N, M, D]
            query = hi.unsqueeze(1)                            # [N, 1, D]
            for layer in self.spatial_layers:
                query = layer(query, context=surr_context)
            TS = query.squeeze(1)                              # [N, D]
        else:
            TS = hi

        return TS


# ---------------------------------------------------------------------------
# ST Path: Spatial → Temporal
# ---------------------------------------------------------------------------

class SpatialTemporalPath(nn.Module):
    """
    Spatial-Temporal Path (Section 3.2.2):
    1. Cross-attention over surrounding vessels at each time step
    2. Self-attention over time series (with learnable aggregation token)
    Paper Eq. 15-23
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.d_model   = d_model
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        self.spatial_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=True)
            for _ in range(num_layers)
        ])
        self.temporal_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=False)
            for _ in range(num_layers)
        ])

    def forward(self, zd, zs_gate, type_emb, surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        Same signature as TemporalSpatialPath.
        Returns: ST [N, D]
        """
        N, T, D = zd.shape
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        # Step 1: Spatial cross-attention at each time step (Eq. 15-19)
        if M > 0:
            spatial_outputs = []
            for t in range(T):
                zt           = zd[:, t, :]                      # [N, D]
                zt_gated     = zt * zs_gate                     # [N, D]
                zt_query     = zt_gated.unsqueeze(1)            # [N, 1, D]
                surr_t       = surrounding_zd[:, :, t, :]       # [N, M, D]
                surr_t_gated = surr_t * surrounding_zs_gate     # [N, M, D]
                st_t = zt_query
                for layer in self.spatial_layers:
                    st_t = layer(st_t, context=surr_t_gated)
                spatial_outputs.append(st_t)                    # [N, 1, D]
            s = torch.cat(spatial_outputs, dim=1)               # [N, T, D]
        else:
            s = zd * zs_gate.unsqueeze(1)                       # [N, T, D]

        # Step 2: Temporal self-attention with aggregation token (Eq. 20-23)
        agg   = self.agg_token.expand(N, -1, -1)
        s_hat = torch.cat([s, agg], dim=1)          # [N, T+1, D]
        s_hat = self.pos_enc(s_hat)

        h = s_hat
        for layer in self.temporal_layers:
            h = layer(h)

        ST = h[:, -1, :]  # [N, D]
        return ST


# ---------------------------------------------------------------------------
# LSTM Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder (Section 3.3, Eq. 24-33).
    Input: fused dual-path features E = concat(TS, ST)
    Output: predicted [lon, lat, v_lon, v_lat, θ_lon, θ_lat] per step
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, pred_len=5, dropout=0.1):
        super().__init__()

        self.pred_len   = pred_len
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.mlp_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.mlp_vel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.mlp_heading = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, E):
        """
        E: [N, input_dim]
        Returns: pos, vel, heading — each [N, pred_len, 2]
        """
        e = self.input_proj(E)
        e = e.unsqueeze(1).expand(-1, self.pred_len, -1)
        h, _ = self.lstm(e)
        pos     = self.mlp_pos(h)
        vel     = self.mlp_vel(h)
        heading = self.mlp_heading(h)
        return pos, vel, heading


# ---------------------------------------------------------------------------
# DualSTMA Full Model
# ---------------------------------------------------------------------------

class DualSTMA(nn.Module):
    """
    DualSTMA: Dual Spatial-Temporal Multi-head Attention
    Huang et al., JMSE 2024

    Hyperparameters (Section 4.1.1):
      d_model=32, hidden_dim=128, num_heads=8, num_layers=4
      dropout=0.1, lstm_hidden=64, lstm_layers=2

    Embedding sizes (verified from dataset):
      num_vessel_types=101  (max vessel_type=100)
      num_lengths=32        (max vessel_length=31)
      num_widths=25         (max vessel_width=24)
    """

    def __init__(self,
                 d_model=32,
                 hidden_dim=128,
                 num_heads=8,
                 num_layers=4,
                 dropout=0.1,
                 lstm_hidden=64,
                 lstm_layers=2,
                 pred_len=5,
                 num_vessel_types=101,
                 num_lengths=32,
                 num_widths=25,
                 type_embed_dim=8):
        super().__init__()

        self.d_model    = d_model
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        self.dynamic_encoder = DynamicFeatureEncoder(
            in_channels=8, d_model=d_model
        )
        self.static_encoder = StaticFeatureEncoder(
            num_vessel_types=num_vessel_types,
            num_lengths=num_lengths,
            num_widths=num_widths,
            type_embed_dim=type_embed_dim,
            d_model=d_model
        )

        self.feature_proj = nn.Linear(d_model, hidden_dim)
        self.gate_proj    = nn.Linear(d_model, hidden_dim)

        self.ts_path = TemporalSpatialPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout
        )
        self.st_path = SpatialTemporalPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout
        )

        self.decoder = LSTMDecoder(
            input_dim=2 * hidden_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            pred_len=pred_len,
            dropout=dropout
        )

    def forward(self, target_dynamic, target_static,
                surrounding_dynamic=None, surrounding_static=None):
        """
        Args:
            target_dynamic:      [N, T, 8]
            target_static:       (type[N], length[N], width[N])
            surrounding_dynamic: [N, M, T, 8] or None
            surrounding_static:  (type[N,M], length[N,M], width[N,M]) or None
        Returns:
            pos:     [N, pred_len, 2]
            vel:     [N, pred_len, 2]
            heading: [N, pred_len, 2]
        """
        N = target_dynamic.shape[0]

        # Encode target
        zd = self.dynamic_encoder(target_dynamic)  # [N, T, d_model]
        zd = self.feature_proj(zd)                  # [N, T, hidden_dim]

        v_type, v_len, v_wid = target_static
        zs_gate, type_emb = self.static_encoder(v_type, v_len, v_wid)
        zs_gate = self.gate_proj(zs_gate)            # [N, hidden_dim]

        # Encode surrounding vessels
        surr_zd   = None
        surr_gate = None
        surr_temb = None

        if surrounding_dynamic is not None and surrounding_dynamic.shape[1] > 0:
            M = surrounding_dynamic.shape[1]
            surr_flat    = surrounding_dynamic.view(N * M, *surrounding_dynamic.shape[2:])
            surr_zd_flat = self.dynamic_encoder(surr_flat)
            surr_zd_flat = self.feature_proj(surr_zd_flat)
            surr_zd      = surr_zd_flat.view(N, M, *surr_zd_flat.shape[1:])

            if surrounding_static is not None:
                s_type, s_len, s_wid = surrounding_static
                s_type_flat = s_type.view(N * M)
                s_len_flat  = s_len.view(N * M)
                s_wid_flat  = s_wid.view(N * M)
                surr_gate_flat, surr_temb_flat = self.static_encoder(
                    s_type_flat, s_len_flat, s_wid_flat
                )
                surr_gate_flat = self.gate_proj(surr_gate_flat)
                surr_gate = surr_gate_flat.view(N, M, -1)
                surr_temb = surr_temb_flat.view(N, M, -1)
            else:
                surr_gate = torch.ones(N, M, self.hidden_dim, device=zd.device)
                surr_temb = torch.zeros(N, M, type_emb.shape[-1], device=zd.device)

        # Dual path encoding
        TS = self.ts_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)
        ST = self.st_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)

        # Fuse and decode
        E = torch.cat([TS, ST], dim=-1)  # [N, 2*hidden_dim]
        pos, vel, heading = self.decoder(E)

        return pos, vel, heading