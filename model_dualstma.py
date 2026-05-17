"""
model_dualstma.py

DualSTMA: Dual Spatial-Temporal Multi-head Attention
Huang et al., Journal of Marine Science and Engineering, 2024
DOI: 10.3390/jmse12112031

Fixes from checklist:
 1. Rotation R_tn by heading θi_tn for target + surrounding vessels
 2. Velocity/heading components computed AFTER rotation
 3. Inverse position (R_tn + P_tn) applied to output
 4. Inverse heading (R_tn) applied to output
 5. Inverse velocity (R_tn) applied to output
 6. MLP order: (type_emb, width, length) — not (type_emb, length, width)
 7. TS path: concat(hd_i, type_emb) + gating for EVERY vessel
 8. TS path: surrounding vessels use same concat+gating for K,V
 9. MLP after each attention module (in addition to FFN)
10. ST path: surrounding vessels use same concat+gating for K,V
11. Heading loss with floormod instead of atan2
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Vessel-Centered Coordinate Transform (Section 3.1.2)
# ---------------------------------------------------------------------------

def rotate_translate(coords, heading_rad, origin):
    """
    Apply vessel-centered transform:
    1. Translate: subtract origin (last observed position)
    2. Rotate: by -heading (so heading becomes positive x-axis)

    coords:      [..., 2] — (lon, lat)
    heading_rad: scalar or [...] — heading in radians
    origin:      [..., 2] — last observed (lon, lat)

    Returns: [..., 2] — transformed coordinates
    """
    translated = coords - origin
    cos_h = torch.cos(heading_rad)
    sin_h = torch.sin(heading_rad)

    lon_t = translated[..., 0]
    lat_t = translated[..., 1]

    # Rotation by -heading (inverse rotation to align heading with x-axis)
    lon_r =  cos_h * lon_t + sin_h * lat_t
    lat_r = -sin_h * lon_t + cos_h * lat_t

    return torch.stack([lon_r, lat_r], dim=-1)


def inverse_rotate_translate(coords, heading_rad, origin):
    """
    Inverse vessel-centered transform:
    1. Inverse rotate: by +heading
    2. Inverse translate: add origin

    coords:      [..., 2]
    heading_rad: scalar or [...]
    origin:      [..., 2]
    """
    cos_h = torch.cos(heading_rad)
    sin_h = torch.sin(heading_rad)

    lon_r = coords[..., 0]
    lat_r = coords[..., 1]

    # Rotation by +heading
    lon_t = cos_h * lon_r - sin_h * lat_r
    lat_t = sin_h * lon_r + cos_h * lat_r

    return torch.stack([lon_t, lat_t], dim=-1) + origin


def rotate_vector(vec, heading_rad):
    """
    Rotate a 2D vector by heading_rad (for velocity and heading vectors).
    vec: [..., 2]
    """
    cos_h = torch.cos(heading_rad)
    sin_h = torch.sin(heading_rad)

    vx = vec[..., 0]
    vy = vec[..., 1]

    vx_r = cos_h * vx - sin_h * vy
    vy_r = sin_h * vx + cos_h * vy

    return torch.stack([vx_r, vy_r], dim=-1)


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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Feature Preprocessing
# ---------------------------------------------------------------------------

class DynamicFeatureEncoder(nn.Module):
    """
    Encodes dynamic features per time step.
    Input: 8 channels [lon, lat, dlon, dlat, v_lon, v_lat, θ_lon, θ_lat]
    — all computed AFTER vessel-centered rotation transform
    Output: [N, T, d_model]
    Paper Eq. 1-3: Conv2d(1x1) + affine (γ,β) + ReLU
    """

    def __init__(self, in_channels=8, d_model=32):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.relu  = nn.ReLU()

    def forward(self, x):
        """x: [N, T, 8] → [N, T, d_model]"""
        x = x.permute(0, 2, 1).unsqueeze(-1)  # [N, 8, T, 1]
        x = self.conv(x)                        # [N, d_model, T, 1]
        x = x.squeeze(-1).permute(0, 2, 1)      # [N, T, d_model]
        x = x * self.gamma + self.beta
        x = self.relu(x)
        return x


class StaticFeatureEncoder(nn.Module):
    """
    Encodes static vessel features → gating signal in [0,1].
    Paper Eq. 4-5: Embedding + MLP(type_emb, width, length) + Sigmoid

    FIX #6: MLP input order is (type_embedding, width, length)
    FIX #6: num_vessel_types=101, num_widths=25, num_lengths=32
    """

    def __init__(self,
                 num_vessel_types=101,
                 num_lengths=32,
                 num_widths=25,
                 type_embed_dim=8,
                 d_model=32):
        super().__init__()

        self.type_embedding = nn.Embedding(num_vessel_types, type_embed_dim)

        # FIX #6: order is (type_emb, width, length)
        mlp_in = type_embed_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, vessel_type, vessel_width, vessel_length):
        """
        FIX #6: width before length (matches paper Eq. 4)
        Returns: gating [N, d_model], type_emb [N, type_embed_dim]
        """
        type_emb = self.type_embedding(vessel_type)    # [N, type_embed_dim]
        width    = vessel_width.float().unsqueeze(-1)   # [N, 1]
        length   = vessel_length.float().unsqueeze(-1)  # [N, 1]

        # FIX #6: (type_emb, width, length) — not (type_emb, length, width)
        z = torch.cat([type_emb, width, length], dim=-1)
        z = self.mlp(z)
        z = self.sigmoid(z)
        return z, type_emb


# ---------------------------------------------------------------------------
# Transformer Building Blocks
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Single Transformer layer: Multi-Head Attention + Add&Norm + FFN + Add&Norm
    FIX #9: Additional MLP after attention module (per paper description)
    Used for both self-attention (temporal) and cross-attention (spatial).
    """

    def __init__(self, d_model=128, num_heads=8, dropout=0.1, cross_attention=False):
        super().__init__()
        self.cross_attention = cross_attention

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)  # FIX #9: extra norm for MLP

        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Standard FFN inside transformer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # FIX #9: Additional MLP after attention module
        self.post_attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, key_padding_mask=None):
        """
        x:       [B, T, d_model] — query
        context: [B, S, d_model] — key/value for cross-attention
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

        # FIX #9: MLP after attention (Add & Norm)
        x = x + self.dropout(self.post_attn_mlp(self.norm3(x)))

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# ---------------------------------------------------------------------------
# Vessel Feature Fusion (used in both TS and ST paths)
# ---------------------------------------------------------------------------

def fuse_vessel_features(hd, type_emb, zs_gate, proj_layer):
    """
    FIX #7, #8: concat(hd, type_emb) → project → gate with zs_gate
    Applied to EVERY vessel (target and surrounding).

    Paper Eq. 10-11 (TS path) and Eq. 15-16 (ST path):
      h_hat = concat(hd, type_emb)
      h     = h_hat ⊙ zs_gate

    hd:       [..., d_model]
    type_emb: [..., type_dim]
    zs_gate:  [..., d_model]
    proj_layer: Linear(d_model + type_dim → d_model)

    Returns: [..., d_model]
    """
    h_hat = torch.cat([hd, type_emb], dim=-1)  # [..., d_model + type_dim]
    h_hat = proj_layer(h_hat)                   # [..., d_model]
    h     = h_hat * zs_gate                     # [..., d_model]
    return h


# ---------------------------------------------------------------------------
# TS Path: Temporal → Spatial
# ---------------------------------------------------------------------------

class TemporalSpatialPath(nn.Module):
    """
    Temporal-Spatial Path (Section 3.2.1, Eq. 6-14):
    1. Temporal self-attention over T time steps (with aggregation token)
    2. Fuse with type_emb + gate (Eq. 10-11) for target AND surrounding
    3. Spatial cross-attention: target as Q, surrounding as K,V

    FIX #7: concat(hd_i, type_emb_i) + gating for target vessel
    FIX #8: concat(hd_j, type_emb_j) + gating for surrounding vessels (K,V)
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4,
                 dropout=0.1, type_embed_dim=8):
        super().__init__()

        self.d_model = d_model

        # Learnable aggregation token (Eq. 6)
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Temporal self-attention layers
        self.temporal_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=False)
            for _ in range(num_layers)
        ])

        # Spatial cross-attention layers
        self.spatial_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=True)
            for _ in range(num_layers)
        ])

        # FIX #7,#8: projection for concat(hd, type_emb) → d_model
        self.fuse_proj = nn.Linear(d_model + type_embed_dim, d_model)

    def _encode_temporal(self, zd):
        """
        Temporal self-attention with aggregation token.
        zd: [N, T, d_model]
        Returns: [N, d_model] — aggregation token output
        """
        N = zd.shape[0]
        agg    = self.agg_token.expand(N, -1, -1)   # [N, 1, d_model]
        zd_hat = torch.cat([zd, agg], dim=1)          # [N, T+1, d_model]
        zd_hat = self.pos_enc(zd_hat)

        h = zd_hat
        for layer in self.temporal_layers:
            h = layer(h)

        return h[:, -1, :]  # [N, d_model] — Eq. 9

    def forward(self, zd, zs_gate, type_emb,
                surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        zd:                   [N, T, d_model]
        zs_gate:              [N, d_model]
        type_emb:             [N, type_dim]
        surrounding_zd:       [N, M, T, d_model] or None
        surrounding_zs_gate:  [N, M, d_model] or None
        surrounding_type_emb: [N, M, type_dim] or None
        Returns: TS [N, d_model]
        """
        N = zd.shape[0]

        # Step 1: Temporal self-attention for target (Eq. 6-9)
        hd_i = self._encode_temporal(zd)  # [N, d_model]

        # FIX #7: concat(hd_i, type_emb_i) + gating (Eq. 10-11)
        hi = fuse_vessel_features(hd_i, type_emb, zs_gate, self.fuse_proj)  # [N, d_model]

        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        if M > 0:
            # FIX #8: temporal encode + concat+gating for each surrounding vessel
            surr_summaries = []
            for m in range(M):
                surr_zd_m    = surrounding_zd[:, m, :, :]       # [N, T, d_model]
                surr_gate_m  = surrounding_zs_gate[:, m, :]     # [N, d_model]
                surr_temb_m  = surrounding_type_emb[:, m, :]    # [N, type_dim]

                # Temporal encode
                hd_j = self._encode_temporal(surr_zd_m)          # [N, d_model]

                # FIX #8: concat(hd_j, type_emb_j) + gating
                hj = fuse_vessel_features(
                    hd_j, surr_temb_m, surr_gate_m, self.fuse_proj
                )  # [N, d_model]
                surr_summaries.append(hj.unsqueeze(1))

            surr_context = torch.cat(surr_summaries, dim=1)  # [N, M, d_model]

            # Spatial cross-attention: target=Q, surrounding=K,V (Eq. 12-14)
            query = hi.unsqueeze(1)  # [N, 1, d_model]
            for layer in self.spatial_layers:
                query = layer(query, context=surr_context)
            TS = query.squeeze(1)    # [N, d_model]
        else:
            TS = hi

        return TS


# ---------------------------------------------------------------------------
# ST Path: Spatial → Temporal
# ---------------------------------------------------------------------------

class SpatialTemporalPath(nn.Module):
    """
    Spatial-Temporal Path (Section 3.2.2, Eq. 15-23):
    1. At each time step: concat+gate for target AND surrounding, then cross-attention
    2. Temporal self-attention with aggregation token

    FIX #10: surrounding vessels use same concat+gating for K,V at each time step
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4,
                 dropout=0.1, type_embed_dim=8):
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

        # FIX #10: projection for concat(zd_t, type_emb) → d_model
        self.fuse_proj = nn.Linear(d_model + type_embed_dim, d_model)

    def forward(self, zd, zs_gate, type_emb,
                surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        Same signature as TemporalSpatialPath.
        Returns: ST [N, d_model]
        """
        N, T, D = zd.shape
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        # Step 1: Spatial cross-attention at each time step (Eq. 15-19)
        if M > 0:
            spatial_outputs = []
            for t in range(T):
                # FIX #10: target at time t — concat+gating (Eq. 15-16)
                zd_t = zd[:, t, :]  # [N, d_model]
                zt_i = fuse_vessel_features(
                    zd_t, type_emb, zs_gate, self.fuse_proj
                )  # [N, d_model]
                zt_query = zt_i.unsqueeze(1)  # [N, 1, d_model]

                # FIX #10: surrounding at time t — same concat+gating for K,V
                surr_t_list = []
                for m in range(M):
                    zd_j_t    = surrounding_zd[:, m, t, :]       # [N, d_model]
                    gate_j    = surrounding_zs_gate[:, m, :]      # [N, d_model]
                    temb_j    = surrounding_type_emb[:, m, :]     # [N, type_dim]
                    zt_j = fuse_vessel_features(
                        zd_j_t, temb_j, gate_j, self.fuse_proj
                    )  # [N, d_model]
                    surr_t_list.append(zt_j.unsqueeze(1))

                surr_t_context = torch.cat(surr_t_list, dim=1)  # [N, M, d_model]

                st_t = zt_query
                for layer in self.spatial_layers:
                    st_t = layer(st_t, context=surr_t_context)
                spatial_outputs.append(st_t)  # [N, 1, d_model]

            s = torch.cat(spatial_outputs, dim=1)  # [N, T, d_model]
        else:
            # No surrounding vessels — just apply gating
            s = fuse_vessel_features(
                zd.view(N * T, D),
                type_emb.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1),
                zs_gate.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1),
                self.fuse_proj
            ).view(N, T, D)

        # Step 2: Temporal self-attention with aggregation token (Eq. 20-23)
        agg   = self.agg_token.expand(N, -1, -1)
        s_hat = torch.cat([s, agg], dim=1)   # [N, T+1, d_model]
        s_hat = self.pos_enc(s_hat)

        h = s_hat
        for layer in self.temporal_layers:
            h = layer(h)

        ST = h[:, -1, :]  # [N, d_model] — Eq. 23
        return ST


# ---------------------------------------------------------------------------
# LSTM Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder (Section 3.3, Eq. 24-33).
    Input: E = concat(TS, ST) [N, 2*d_model]
    Output: pos, vel, heading — each [N, pred_len, 2]

    Inverse transform (R_tn, P_tn) applied OUTSIDE this module
    to all three outputs (position, velocity, heading).
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2,
                 pred_len=5, dropout=0.1):
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

        # Three separate MLPs (Eq. 31-33)
        self.mlp_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # lon, lat
        )
        self.mlp_vel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # v_lon, v_lat
        )
        self.mlp_heading = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # θ_lon, θ_lat
        )

    def forward(self, E):
        """
        E: [N, input_dim]
        Returns: pos, vel, heading — each [N, pred_len, 2]
        """
        e = self.input_proj(E)
        e = e.unsqueeze(1).expand(-1, self.pred_len, -1)  # [N, pred_len, hidden]
        h, _ = self.lstm(e)

        pos     = self.mlp_pos(h)      # [N, pred_len, 2] — in vessel-centered space
        vel     = self.mlp_vel(h)      # [N, pred_len, 2]
        heading = self.mlp_heading(h)  # [N, pred_len, 2]

        return pos, vel, heading


# ---------------------------------------------------------------------------
# DualSTMA Full Model
# ---------------------------------------------------------------------------

class DualSTMA(nn.Module):
    """
    DualSTMA: Dual Spatial-Temporal Multi-head Attention
    Huang et al., JMSE 2024

    Hyperparameters (Section 4.1.1):
      d_model=32, hidden_dim=128, num_heads=8, num_layers=4, dropout=0.1
      lstm_hidden=64, lstm_layers=2
      lr=0.0015, batch=256, epochs=100, Adam

    Embedding sizes (verified from dataset):
      num_vessel_types=101, num_lengths=32, num_widths=25
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

        self.d_model       = d_model
        self.hidden_dim    = hidden_dim
        self.pred_len      = pred_len
        self.type_embed_dim = type_embed_dim

        # Feature encoders
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

        # Project d_model → hidden_dim for Transformer paths
        self.feature_proj = nn.Linear(d_model, hidden_dim)
        self.gate_proj    = nn.Linear(d_model, hidden_dim)
        self.temb_proj    = nn.Linear(type_embed_dim, type_embed_dim)  # passthrough

        # Dual encoder paths
        self.ts_path = TemporalSpatialPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
            type_embed_dim=type_embed_dim
        )
        self.st_path = SpatialTemporalPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
            type_embed_dim=type_embed_dim
        )

        # LSTM Decoder
        self.decoder = LSTMDecoder(
            input_dim=2 * hidden_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            pred_len=pred_len,
            dropout=dropout
        )

    def forward(self,
                target_dynamic,       # [N, T, 8] — already in vessel-centered space
                target_static,        # (type[N], width[N], length[N])
                heading_rad,          # [N] — heading at last obs (for inverse transform)
                last_obs_pos,         # [N, 2] — last obs position (for inverse transform)
                surrounding_dynamic=None,   # [N, M, T, 8] or None
                surrounding_static=None):   # (type[N,M], width[N,M], length[N,M]) or None
        """
        Returns:
            pos:     [N, pred_len, 2] — predicted (lon, lat) in ORIGINAL space
            vel:     [N, pred_len, 2] — predicted velocity in ORIGINAL space
            heading: [N, pred_len, 2] — predicted heading in ORIGINAL space
        """
        N = target_dynamic.shape[0]

        # --- Encode target vessel ---
        zd = self.dynamic_encoder(target_dynamic)  # [N, T, d_model]
        zd = self.feature_proj(zd)                  # [N, T, hidden_dim]

        v_type, v_width, v_length = target_static
        zs_gate, type_emb = self.static_encoder(v_type, v_width, v_length)
        zs_gate  = self.gate_proj(zs_gate)           # [N, hidden_dim]
        # type_emb stays as [N, type_embed_dim]

        # --- Encode surrounding vessels ---
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
                s_type, s_width, s_length = surrounding_static
                s_type_flat   = s_type.view(N * M)
                s_width_flat  = s_width.view(N * M)
                s_length_flat = s_length.view(N * M)

                surr_gate_flat, surr_temb_flat = self.static_encoder(
                    s_type_flat, s_width_flat, s_length_flat
                )
                surr_gate_flat = self.gate_proj(surr_gate_flat)
                surr_gate = surr_gate_flat.view(N, M, -1)   # [N, M, hidden_dim]
                surr_temb = surr_temb_flat.view(N, M, -1)   # [N, M, type_embed_dim]
            else:
                surr_gate = torch.ones(N, M, self.hidden_dim, device=zd.device)
                surr_temb = torch.zeros(N, M, self.type_embed_dim, device=zd.device)

        # --- Dual path encoding ---
        TS = self.ts_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)
        ST = self.st_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)

        # --- Fuse and decode (Eq. 24) ---
        E = torch.cat([TS, ST], dim=-1)              # [N, 2*hidden_dim]
        pos_vc, vel_vc, heading_vc = self.decoder(E) # all in vessel-centered space

        # --- FIX #3,#4,#5: Inverse transform (R_tn, P_tn) ---
        # heading_rad: [N] → [N, pred_len] for broadcasting
        head_expanded = heading_rad.unsqueeze(1).expand(-1, self.pred_len)  # [N, pred_len]
        origin_expanded = last_obs_pos.unsqueeze(1).expand(-1, self.pred_len, -1)  # [N, pred_len, 2]

        # Position: inverse rotate + translate
        pos = inverse_rotate_translate(pos_vc, head_expanded, origin_expanded)

        # Velocity: inverse rotate only
        vel = rotate_vector(vel_vc, head_expanded)

        # Heading: inverse rotate only
        heading = rotate_vector(heading_vc, head_expanded)

        return pos, vel, heading