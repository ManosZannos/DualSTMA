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
 6. MLP order: (type_emb, width, length)
 7. TS path: concat(hd_i, type_emb) + gating for EVERY vessel
 8. TS path: surrounding vessels use same concat+gating for K,V
 9. MLP after each attention module
10. ST path: surrounding vessels use same concat+gating for K,V
11. Heading loss with floormod instead of atan2
12. [v3] Padding mask for surrounding vessels in spatial attention (Eq. 13, 18)
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
    Apply vessel-centered transform.
    coords:      [..., 2]
    heading_rad: [...] or scalar
    origin:      [..., 2]
    """
    translated = coords - origin
    cos_h = torch.cos(heading_rad)
    sin_h = torch.sin(heading_rad)
    lon_t = translated[..., 0]
    lat_t = translated[..., 1]
    lon_r =  cos_h * lon_t + sin_h * lat_t
    lat_r = -sin_h * lon_t + cos_h * lat_t
    return torch.stack([lon_r, lat_r], dim=-1)


def inverse_rotate_translate(coords, heading_rad, origin):
    """
    Inverse vessel-centered transform.
    coords:      [..., 2]
    heading_rad: [...] or scalar
    origin:      [..., 2]
    """
    cos_h = torch.cos(heading_rad)
    sin_h = torch.sin(heading_rad)
    lon_r = coords[..., 0]
    lat_r = coords[..., 1]
    lon_t = cos_h * lon_r - sin_h * lat_r
    lat_t = sin_h * lon_r + cos_h * lat_r
    return torch.stack([lon_t, lat_t], dim=-1) + origin


def rotate_vector(vec, heading_rad):
    """Rotate a 2D vector by heading_rad. vec: [..., 2]"""
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Feature Preprocessing
# ---------------------------------------------------------------------------

class DynamicFeatureEncoder(nn.Module):
    """
    Conv2d(1x1) + affine (γ,β) + ReLU (Eq. 1-3)
    Input: [N, T, 8] → Output: [N, T, d_model]
    """
    def __init__(self, in_channels=8, d_model=32):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv(x)
        x = x.squeeze(-1).permute(0, 2, 1)
        x = x * self.gamma + self.beta
        x = self.relu(x)
        return x


class StaticFeatureEncoder(nn.Module):
    """
    MLP(type_embedding, width, length) + Sigmoid (Eq. 4-5)
    FIX #6: order is (type_emb, width, length)
    """
    def __init__(self, num_vessel_types=101, num_lengths=32, num_widths=25,
                 type_embed_dim=8, d_model=32):
        super().__init__()
        self.type_embedding = nn.Embedding(num_vessel_types, type_embed_dim)
        mlp_in = type_embed_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, vessel_type, vessel_width, vessel_length):
        type_emb = self.type_embedding(vessel_type)
        width    = vessel_width.float().unsqueeze(-1)
        length   = vessel_length.float().unsqueeze(-1)
        z = torch.cat([type_emb, width, length], dim=-1)
        z = self.mlp(z)
        z = self.sigmoid(z)
        return z, type_emb


# ---------------------------------------------------------------------------
# Transformer Building Blocks
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """
    Multi-Head Attention + Add&Norm + MLP + FFN + Add&Norm
    FIX #9: MLP after attention module
    """
    def __init__(self, d_model=128, num_heads=8, dropout=0.1, cross_attention=False):
        super().__init__()
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
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
        self.post_attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, key_padding_mask=None):
        """
        x:               [B, T, d_model]
        context:         [B, S, d_model] — for cross-attention
        key_padding_mask:[B, S] bool — True = ignore (padded vessel)

        POST-NORM (paper: "apply layer normalization after each attention module"):
          x → attention → Add → Norm → MLP → Add → Norm → FFN → Add → Norm
        """
        if self.cross_attention and context is not None:
            # FIX: if all surrounding vessels are padded for a sample,
            # set that sample's mask to all False to avoid NaN in softmax
            safe_mask = key_padding_mask
            if key_padding_mask is not None:
                all_masked = key_padding_mask.all(dim=-1, keepdim=True)  # [B, 1]
                if all_masked.any():
                    safe_mask = key_padding_mask.clone()
                    safe_mask[all_masked.squeeze(-1)] = False
            x2, _ = self.attn(
                x, context, context,
                key_padding_mask=safe_mask
            )
        else:
            x2, _ = self.attn(
                x, x, x,
                key_padding_mask=key_padding_mask
            )
        # Post-norm: Add & Norm after attention
        x = self.norm1(x + self.dropout(x2))
        # Post-norm: Add & Norm after MLP
        x = self.norm3(x + self.dropout(self.post_attn_mlp(x)))
        # Post-norm: Add & Norm after FFN
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ---------------------------------------------------------------------------
# Vessel Feature Fusion
# ---------------------------------------------------------------------------

def fuse_vessel_features(hd, type_emb, zs_gate, proj_layer):
    """
    FIX #7,#8,#10: concat(hd, type_emb) → project → gate
    Eq. 10-11 (TS) and Eq. 15-16 (ST)
    """
    h_hat = torch.cat([hd, type_emb], dim=-1)
    h_hat = proj_layer(h_hat)
    h     = h_hat * zs_gate
    return h


# ---------------------------------------------------------------------------
# TS Path: Temporal → Spatial
# ---------------------------------------------------------------------------

class TemporalSpatialPath(nn.Module):
    """
    Temporal-Spatial Path (Section 3.2.1, Eq. 6-14)
    FIX #12: padding mask for spatial cross-attention
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4,
                 dropout=0.1, type_embed_dim=8):
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
        self.fuse_proj = nn.Linear(d_model + type_embed_dim, d_model)

    def _encode_temporal(self, zd):
        N = zd.shape[0]
        agg    = self.agg_token.expand(N, -1, -1)
        zd_hat = torch.cat([zd, agg], dim=1)
        zd_hat = self.pos_enc(zd_hat)
        h = zd_hat
        for layer in self.temporal_layers:
            h = layer(h)
        return h[:, -1, :]

    def forward(self, zd, zs_gate, type_emb,
                surrounding_zd, surrounding_zs_gate, surrounding_type_emb,
                surr_mask=None):
        """
        surr_mask: [N, M] bool — True = real vessel, False = padded
        FIX #12: convert to key_padding_mask (True = ignore)
        """
        N = zd.shape[0]

        # Target vessel: temporal encode + fuse (Eq. 6-11)
        hd_i = self._encode_temporal(zd)
        hi   = fuse_vessel_features(hd_i, type_emb, zs_gate, self.fuse_proj)

        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        if M > 0:
            # Surrounding vessels: temporal encode + fuse
            surr_summaries = []
            for m in range(M):
                surr_zd_m   = surrounding_zd[:, m, :, :]
                surr_gate_m = surrounding_zs_gate[:, m, :]
                surr_temb_m = surrounding_type_emb[:, m, :]
                hd_j = self._encode_temporal(surr_zd_m)
                hj   = fuse_vessel_features(hd_j, surr_temb_m, surr_gate_m, self.fuse_proj)
                surr_summaries.append(hj.unsqueeze(1))

            surr_context = torch.cat(surr_summaries, dim=1)  # [N, M, d_model]

            # FIX #12: padding mask — True = padded (ignore)
            key_padding_mask = None
            if surr_mask is not None:
                key_padding_mask = ~surr_mask  # [N, M] True=ignore

            # Spatial cross-attention (Eq. 12-14)
            query = hi.unsqueeze(1)
            for layer in self.spatial_layers:
                query = layer(query, context=surr_context,
                              key_padding_mask=key_padding_mask)
            TS = query.squeeze(1)
        else:
            TS = hi

        return TS


# ---------------------------------------------------------------------------
# ST Path: Spatial → Temporal
# ---------------------------------------------------------------------------

class SpatialTemporalPath(nn.Module):
    """
    Spatial-Temporal Path (Section 3.2.2, Eq. 15-23)
    FIX #12: padding mask for spatial cross-attention
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
        self.fuse_proj = nn.Linear(d_model + type_embed_dim, d_model)

    def forward(self, zd, zs_gate, type_emb,
                surrounding_zd, surrounding_zs_gate, surrounding_type_emb,
                surr_mask=None):
        """
        surr_mask: [N, M] bool — True = real vessel, False = padded
        FIX #12: padding mask per time step
        """
        N, T, D = zd.shape
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        # FIX #12: key_padding_mask
        key_padding_mask = None
        if surr_mask is not None and M > 0:
            key_padding_mask = ~surr_mask  # [N, M] True=ignore

        # Step 1: Spatial cross-attention at each time step (Eq. 15-19)
        if M > 0:
            spatial_outputs = []
            for t in range(T):
                zd_t     = zd[:, t, :]
                zt_i     = fuse_vessel_features(zd_t, type_emb, zs_gate, self.fuse_proj)
                zt_query = zt_i.unsqueeze(1)

                surr_t_list = []
                for m in range(M):
                    zd_j_t = surrounding_zd[:, m, t, :]
                    gate_j = surrounding_zs_gate[:, m, :]
                    temb_j = surrounding_type_emb[:, m, :]
                    zt_j   = fuse_vessel_features(zd_j_t, temb_j, gate_j, self.fuse_proj)
                    surr_t_list.append(zt_j.unsqueeze(1))

                surr_t_context = torch.cat(surr_t_list, dim=1)  # [N, M, d_model]

                st_t = zt_query
                for layer in self.spatial_layers:
                    st_t = layer(st_t, context=surr_t_context,
                                 key_padding_mask=key_padding_mask)
                spatial_outputs.append(st_t)

            s = torch.cat(spatial_outputs, dim=1)  # [N, T, d_model]
        else:
            s = fuse_vessel_features(
                zd.view(N * T, D),
                type_emb.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1),
                zs_gate.unsqueeze(1).expand(-1, T, -1).reshape(N * T, -1),
                self.fuse_proj
            ).view(N, T, D)

        # Step 2: Temporal self-attention (Eq. 20-23)
        agg   = self.agg_token.expand(N, -1, -1)
        s_hat = torch.cat([s, agg], dim=1)
        s_hat = self.pos_enc(s_hat)
        h = s_hat
        for layer in self.temporal_layers:
            h = layer(h)
        ST = h[:, -1, :]
        return ST


# ---------------------------------------------------------------------------
# LSTM Decoder
# ---------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """LSTM Decoder (Section 3.3, Eq. 24-33)."""

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
        self.mlp_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.mlp_vel = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self.mlp_heading = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, E):
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

    v3 changes:
    - FIX #12: padding mask for surrounding vessels in spatial attention
    """

    def __init__(self,
                 d_model=32, hidden_dim=128, num_heads=8, num_layers=4,
                 dropout=0.1, lstm_hidden=64, lstm_layers=2, pred_len=5,
                 num_vessel_types=101, num_lengths=32, num_widths=25,
                 type_embed_dim=8):
        super().__init__()

        self.d_model        = d_model
        self.hidden_dim     = hidden_dim
        self.pred_len       = pred_len
        self.type_embed_dim = type_embed_dim

        self.dynamic_encoder = DynamicFeatureEncoder(in_channels=8, d_model=d_model)
        self.static_encoder  = StaticFeatureEncoder(
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
            num_layers=num_layers, dropout=dropout,
            type_embed_dim=type_embed_dim
        )
        self.st_path = SpatialTemporalPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout,
            type_embed_dim=type_embed_dim
        )

        self.decoder = LSTMDecoder(
            input_dim=2 * hidden_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            pred_len=pred_len,
            dropout=dropout
        )

    def forward(self,
                target_dynamic,          # [N, T, 8]
                target_static,           # (type[N], width[N], length[N])
                heading_rad,             # [N]
                last_obs_pos,            # [N, 2]
                surrounding_dynamic=None,  # [N, M, T, 8]
                surrounding_static=None,   # (type[N,M], width[N,M], length[N,M])
                surr_mask=None):           # [N, M] bool — True=real, False=padded
        """
        FIX #12: surr_mask passed to TS and ST paths for padding mask.
        """
        N = target_dynamic.shape[0]

        # Encode target
        zd = self.dynamic_encoder(target_dynamic)
        zd = self.feature_proj(zd)

        v_type, v_width, v_length = target_static
        zs_gate, type_emb = self.static_encoder(v_type, v_width, v_length)
        zs_gate = self.gate_proj(zs_gate)

        # Encode surrounding
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
                surr_gate = surr_gate_flat.view(N, M, -1)
                surr_temb = surr_temb_flat.view(N, M, -1)
            else:
                surr_gate = torch.ones(N, M, self.hidden_dim, device=zd.device)
                surr_temb = torch.zeros(N, M, self.type_embed_dim, device=zd.device)

        # Dual path encoding — FIX #12: pass surr_mask
        TS = self.ts_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb,
                          surr_mask=surr_mask)
        ST = self.st_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb,
                          surr_mask=surr_mask)

        # Fuse and decode
        E = torch.cat([TS, ST], dim=-1)
        pos_vc, vel_vc, heading_vc = self.decoder(E)

        # Inverse transform
        head_expanded   = heading_rad.unsqueeze(1).expand(-1, self.pred_len)
        origin_expanded = last_obs_pos.unsqueeze(1).expand(-1, self.pred_len, -1)

        pos     = inverse_rotate_translate(pos_vc, head_expanded, origin_expanded)
        vel     = rotate_vector(vel_vc, head_expanded)
        heading = rotate_vector(heading_vc, head_expanded)

        return pos, vel, heading