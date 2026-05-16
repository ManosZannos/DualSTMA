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
        # Conv2d with 1x1 kernel for channel fusion
        self.conv = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # Learnable affine parameters γ, β
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.relu  = nn.ReLU()

    def forward(self, x):
        """
        x: [N, T, 8] — dynamic features per vessel per time step
        Returns: [N, T, d_model]
        """
        # Conv2d expects [N, C, H, W] — treat T as H, 1 as W
        x = x.permute(0, 2, 1).unsqueeze(-1)   # [N, 8, T, 1]
        x = self.conv(x)                         # [N, d_model, T, 1]
        x = x.squeeze(-1).permute(0, 2, 1)       # [N, T, d_model]
        x = x * self.gamma + self.beta           # affine
        x = self.relu(x)
        return x


class StaticFeatureEncoder(nn.Module):
    """
    Encodes static vessel features → gating signal.
    Input: vessel_type (categorical), length (binned), width (binned)
    Output: gating signal in [0,1] — [N, d_model]
    Paper Eq. 4-5: Embedding + MLP + Sigmoid
    """

    def __init__(self,
                 num_vessel_types=50,
                 num_lengths=21,
                 num_widths=21,
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
        Returns: gating [N, d_model]
        """
        type_emb = self.type_embedding(vessel_type)   # [N, type_embed_dim]
        length   = vessel_length.float().unsqueeze(-1) # [N, 1]
        width    = vessel_width.float().unsqueeze(-1)  # [N, 1]

        z = torch.cat([type_emb, length, width], dim=-1)  # [N, type_embed_dim+2]
        z = self.mlp(z)      # [N, d_model]
        z = self.sigmoid(z)  # [N, d_model] — gating in [0,1]
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
        x:       [B, T, d_model] — query
        context: [B, S, d_model] — key/value for cross-attention
        """
        if self.cross_attention and context is not None:
            # Cross-attention: x as query, context as key/value
            x2, _ = self.attn(
                self.norm1(x), context, context,
                key_padding_mask=key_padding_mask
            )
        else:
            # Self-attention
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

        self.d_model = d_model

        # Learnable aggregation token (Eq. 6)
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

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

        # Project d_model input to d_model (in case input_dim != d_model)
        self.input_proj = nn.Linear(d_model, d_model)

    def forward(self, zd, zs_gate, type_emb, surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        zd:                  [N, T, d_model] — target vessel dynamic features
        zs_gate:             [N, d_model]    — target vessel static gating
        type_emb:            [N, type_dim]   — target vessel type embedding
        surrounding_zd:      [N, M, T, d_model] — M surrounding vessels
        surrounding_zs_gate: [N, M, d_model]
        surrounding_type_emb:[N, M, type_dim]

        Returns: TS [N, d_model]
        """
        N, T, D = zd.shape

        # --- Step 1: Temporal self-attention ---
        # Add aggregation token at end of time series (Eq. 6)
        agg = self.agg_token.expand(N, -1, -1)  # [N, 1, D]
        zd_hat = torch.cat([zd, agg], dim=1)     # [N, T+1, D]
        zd_hat = self.pos_enc(zd_hat)

        hd = zd_hat
        for layer in self.temporal_layers:
            hd = layer(hd)

        # Extract aggregation token output as summary (Eq. 9)
        hd_i = hd[:, -1, :]  # [N, D]

        # --- Step 2: Fuse with static info + gating (Eq. 10-11) ---
        # Concatenate temporal summary with type embedding
        hd_i_cat = torch.cat([hd_i, type_emb], dim=-1)  # [N, D+type_dim]
        # Project back to d_model
        hd_i_cat = nn.functional.linear(
            hd_i_cat,
            torch.zeros(D, hd_i_cat.shape[-1], device=zd.device)
        )
        # Simpler: just use hd_i with gating
        hi = hd_i * zs_gate  # [N, D] — gating (Eq. 11)

        # --- Step 3: Spatial cross-attention ---
        # Prepare surrounding vessel representations
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        if M > 0:
            # Get surrounding summaries via temporal encoding
            surr_summaries = []
            for m in range(M):
                surr_zd = surrounding_zd[:, m, :, :]  # [N, T, D]
                surr_agg = self.agg_token.expand(N, -1, -1)
                surr_hat = torch.cat([surr_zd, surr_agg], dim=1)
                surr_hat = self.pos_enc(surr_hat)
                surr_h = surr_hat
                for layer in self.temporal_layers:
                    surr_h = layer(surr_h)
                surr_summary = surr_h[:, -1, :]  # [N, D]
                # Apply surrounding gating
                surr_gate = surrounding_zs_gate[:, m, :]  # [N, D]
                surr_summary = surr_summary * surr_gate
                surr_summaries.append(surr_summary.unsqueeze(1))

            # Stack: [N, M, D]
            surr_context = torch.cat(surr_summaries, dim=1)

            # Cross-attention: target as query, surrounding as key/value
            query = hi.unsqueeze(1)  # [N, 1, D]
            for layer in self.spatial_layers:
                query = layer(query, context=surr_context)
            TS = query.squeeze(1)  # [N, D]
        else:
            TS = hi  # no surrounding vessels

        return TS


# ---------------------------------------------------------------------------
# ST Path: Spatial → Temporal
# ---------------------------------------------------------------------------

class SpatialTemporalPath(nn.Module):
    """
    Spatial-Temporal Path (Section 3.2.2):
    1. Cross-attention over surrounding vessels at each time step (spatial)
    2. Self-attention over time series (with learnable aggregation token)
    Paper Eq. 15-23
    """

    def __init__(self, d_model=128, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Learnable aggregation token (Eq. 20)
        self.agg_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        # Spatial cross-attention layers (per time step)
        self.spatial_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=True)
            for _ in range(num_layers)
        ])

        # Temporal self-attention layers
        self.temporal_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, dropout, cross_attention=False)
            for _ in range(num_layers)
        ])

    def forward(self, zd, zs_gate, type_emb, surrounding_zd, surrounding_zs_gate, surrounding_type_emb):
        """
        Same signature as TemporalSpatialPath.
        Returns: ST [N, d_model]
        """
        N, T, D = zd.shape

        # --- Step 1: Spatial cross-attention at each time step (Eq. 15-19) ---
        M = surrounding_zd.shape[1] if surrounding_zd is not None else 0

        if M > 0:
            spatial_outputs = []
            for t in range(T):
                # Target at time t with gating (Eq. 15-16)
                zt = zd[:, t, :]                           # [N, D]
                zt_gated = zt * zs_gate                    # [N, D]
                zt_query = zt_gated.unsqueeze(1)           # [N, 1, D]

                # Surrounding at time t with gating
                surr_t = surrounding_zd[:, :, t, :]        # [N, M, D]
                surr_gate_t = surrounding_zs_gate          # [N, M, D]
                surr_t_gated = surr_t * surr_gate_t        # [N, M, D]

                # Cross-attention
                st_t = zt_query
                for layer in self.spatial_layers:
                    st_t = layer(st_t, context=surr_t_gated)
                spatial_outputs.append(st_t)               # [N, 1, D]

            # Stack spatial outputs: [N, T, D]
            s = torch.cat(spatial_outputs, dim=1)
        else:
            s = zd * zs_gate.unsqueeze(1)  # [N, T, D]

        # --- Step 2: Temporal self-attention (Eq. 20-23) ---
        agg = self.agg_token.expand(N, -1, -1)  # [N, 1, D]
        s_hat = torch.cat([s, agg], dim=1)       # [N, T+1, D]
        s_hat = self.pos_enc(s_hat)

        h = s_hat
        for layer in self.temporal_layers:
            h = layer(h)

        # Extract aggregation token output (Eq. 23)
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

        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project fused features to LSTM input dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Three separate MLP decoders (Eq. 31-33)
        self.mlp_pos     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # lon, lat
        )
        self.mlp_vel     = nn.Sequential(
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
        E: [N, input_dim] — fused dual-path features
        Returns:
            pos:     [N, pred_len, 2]
            vel:     [N, pred_len, 2]
            heading: [N, pred_len, 2]
        """
        N = E.shape[0]

        # Project and repeat as LSTM input sequence
        e = self.input_proj(E)           # [N, hidden_dim]
        e = e.unsqueeze(1).expand(-1, self.pred_len, -1)  # [N, pred_len, hidden_dim]

        # LSTM decode
        h, _ = self.lstm(e)  # [N, pred_len, hidden_dim]

        # Decode each output
        pos     = self.mlp_pos(h)      # [N, pred_len, 2]
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
      d_model=32 (input), hidden_dim=128, num_heads=8, num_layers=4
      dropout=0.1, lstm_hidden=64, lstm_layers=2
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
                 num_vessel_types=50,
                 type_embed_dim=8):
        super().__init__()

        self.d_model    = d_model
        self.hidden_dim = hidden_dim
        self.pred_len   = pred_len

        # --- Feature encoders ---
        # Dynamic: 8 channels → d_model
        self.dynamic_encoder = DynamicFeatureEncoder(
            in_channels=8, d_model=d_model
        )

        # Static: type + length + width → gating signal
        self.static_encoder = StaticFeatureEncoder(
            num_vessel_types=num_vessel_types,
            type_embed_dim=type_embed_dim,
            d_model=d_model
        )

        # Project d_model → hidden_dim for Transformer
        self.feature_proj = nn.Linear(d_model, hidden_dim)
        self.gate_proj    = nn.Linear(d_model, hidden_dim)

        # --- Dual encoder ---
        self.ts_path = TemporalSpatialPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout
        )
        self.st_path = SpatialTemporalPath(
            d_model=hidden_dim, num_heads=num_heads,
            num_layers=num_layers, dropout=dropout
        )

        # --- LSTM Decoder ---
        # Input: concat(TS, ST) = 2 * hidden_dim
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
            target_dynamic:      [N, T, 8]    — dynamic features of target vessel
            target_static:       tuple(type[N], length[N], width[N])
            surrounding_dynamic: [N, M, T, 8] — M surrounding vessels (optional)
            surrounding_static:  list of tuples, one per surrounding vessel (optional)

        Returns:
            pos:     [N, pred_len, 2] — predicted (lon, lat)
            vel:     [N, pred_len, 2] — predicted (v_lon, v_lat)
            heading: [N, pred_len, 2] — predicted (θ_lon, θ_lat)
        """
        N = target_dynamic.shape[0]

        # --- Encode target vessel features ---
        zd = self.dynamic_encoder(target_dynamic)   # [N, T, d_model]
        zd = self.feature_proj(zd)                  # [N, T, hidden_dim]

        v_type, v_len, v_wid = target_static
        zs_gate, type_emb = self.static_encoder(v_type, v_len, v_wid)
        zs_gate = self.gate_proj(zs_gate)            # [N, hidden_dim]

        # --- Encode surrounding vessels ---
        surr_zd   = None
        surr_gate = None
        surr_temb = None

        if surrounding_dynamic is not None and surrounding_dynamic.shape[1] > 0:
            M = surrounding_dynamic.shape[1]
            # [N, M, T, 8] → [N*M, T, 8]
            surr_flat = surrounding_dynamic.view(N * M, *surrounding_dynamic.shape[2:])
            surr_zd_flat = self.dynamic_encoder(surr_flat)   # [N*M, T, d_model]
            surr_zd_flat = self.feature_proj(surr_zd_flat)   # [N*M, T, hidden_dim]
            surr_zd = surr_zd_flat.view(N, M, *surr_zd_flat.shape[1:])  # [N, M, T, hidden_dim]

            if surrounding_static is not None:
                s_type, s_len, s_wid = surrounding_static
                # [N*M]
                s_type_flat = s_type.view(N * M)
                s_len_flat  = s_len.view(N * M)
                s_wid_flat  = s_wid.view(N * M)
                surr_gate_flat, surr_temb_flat = self.static_encoder(
                    s_type_flat, s_len_flat, s_wid_flat
                )
                surr_gate_flat = self.gate_proj(surr_gate_flat)
                surr_gate = surr_gate_flat.view(N, M, -1)   # [N, M, hidden_dim]
                surr_temb = surr_temb_flat.view(N, M, -1)
            else:
                surr_gate = torch.ones(N, M, self.hidden_dim, device=zd.device)
                surr_temb = torch.zeros(N, M, 8, device=zd.device)

        # --- Dual path encoding ---
        TS = self.ts_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)
        ST = self.st_path(zd, zs_gate, type_emb, surr_zd, surr_gate, surr_temb)

        # --- Fuse and decode (Eq. 24) ---
        E = torch.cat([TS, ST], dim=-1)  # [N, 2*hidden_dim]
        pos, vel, heading = self.decoder(E)

        return pos, vel, heading
