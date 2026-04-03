"""
Hypergraph+LSTM Model

This is using hypredges instead of standard graph, because of the 
public transport naturally a hypergraph like representation.

Feng et al. (2019) - Hypergraph Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
import os


class HypergraphConfig:
    NUM_ZONES       = 1419
    NUM_SCENARIOS   = 3

    # Hypergraph — same as GAT+LSTM
    HG_IN_CHANNELS  = 16
    HG_HIDDEN       = 64
    HG_OUT_CHANNELS = 32
    HG_NUM_LAYERS   = 2
    HG_DROPOUT      = 0.2

    LSTM_INPUT_SIZE  = 32
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS  = 2
    LSTM_DROPOUT     = 0.2

    OUTPUT_SIZE   = NUM_ZONES
    BATCH_SIZE    = 4
    LEARNING_RATE = 1e-3
    NUM_EPOCHS    = 100
    PATIENCE      = 15

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_incidence_matrix(zone_ids: List[int],
                            gtfs_routes: Optional[Dict] = None,
                            gdf=None) -> torch.Tensor:
    """
    H ∈ {0,1}^(N×M) matrix creation.
    N = number of zones, M = number of hyperedges (lines)
    H[i,e] = 1 if i. zone is in the hyperedge.
    """
    n = len(zone_ids)
    zone_to_idx = {z: i for i, z in enumerate(zone_ids)}

    if gtfs_routes is not None:
        print('Hyperedge creation based on GTFS')
        edges = []
        for route_id, stop_zone_ids in gtfs_routes.items():
            valid = [zone_to_idx[z] for z in stop_zone_ids if z in zone_to_idx]
            if len(valid) >= 2:
                edges.append(valid)

        m = len(edges)
        H = torch.zeros(n, m, dtype=torch.float32)
        for e_idx, zone_indices in enumerate(edges):
            for z_idx in zone_indices:
                H[z_idx, e_idx] = 1.0
        print(f'  ✅ GTFS hypredges: {m} lines, {H.sum().int()} connection')

    else:
        print('hyperedge based on distance (GTFS missing)...')
        if gdf is not None:
            centroids = np.column_stack([
                gdf.set_index('NO').reindex(zone_ids)['centroid_lon'].fillna(0),
                gdf.set_index('NO').reindex(zone_ids)['centroid_lat'].fillna(0)
            ])
        else:
            centroids = np.random.rand(n, 2)

        num_edges = max(50, n // 10)
        np.random.seed(42)
        centers = centroids[np.random.choice(n, num_edges, replace=False)]

        H = torch.zeros(n, num_edges, dtype=torch.float32)
        for i, c in enumerate(centroids):
            dists   = np.linalg.norm(centers - c, axis=1)
            nearest = np.argsort(dists)[:3]
            for e in nearest:
                H[i, e] = 1.0
        m = num_edges
        print(f'  ✅ {m} hyperedges based on distance, {H.sum().int()} connection')

    return H


def normalize_incidence(H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HGNN normalization (Feng et al., 2019).
    D_v = zone degree, D_e = hyperedge degree.
    """
    d_v = torch.clamp(H.sum(dim=1), min=1.0)
    d_e = torch.clamp(H.sum(dim=0), min=1.0)
    return H, torch.diag(d_v ** -0.5), torch.diag(d_e ** -1.0)


class HypergraphConv(nn.Module):
    """
    Hypergraph convolution layer — Feng et al. (2019).
    X' = D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X W
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.linear  = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.bn      = nn.BatchNorm1d(out_channels)

    def forward(self, x, H, D_v_inv_sqrt, D_e_inv):
        x = D_v_inv_sqrt @ x
        x = H.T @ x
        x = D_e_inv @ x
        x = H @ x
        x = D_v_inv_sqrt @ x
        x = self.linear(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class HypergraphEncoder(nn.Module):
    """Multi layer hypergraph encoder."""

    def __init__(self, cfg: HypergraphConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            HypergraphConv(cfg.HG_IN_CHANNELS, cfg.HG_HIDDEN, cfg.HG_DROPOUT)
        ])
        for _ in range(cfg.HG_NUM_LAYERS - 2):
            self.layers.append(HypergraphConv(cfg.HG_HIDDEN, cfg.HG_HIDDEN, cfg.HG_DROPOUT))
        self.layers.append(HypergraphConv(cfg.HG_HIDDEN, cfg.HG_OUT_CHANNELS, cfg.HG_DROPOUT))

    def forward(self, x, H, D_v_inv_sqrt, D_e_inv):
        for layer in self.layers:
            x = layer(x, H, D_v_inv_sqrt, D_e_inv)
        return x


class LSTMEncoder(nn.Module):
    """LSTM encoder — same as GAT+LSTM"""

    def __init__(self, cfg: HypergraphConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.NUM_ZONES * cfg.HG_OUT_CHANNELS, cfg.LSTM_INPUT_SIZE * 8)
        self.lstm = nn.LSTM(
            input_size   = cfg.LSTM_INPUT_SIZE * 8,
            hidden_size  = cfg.LSTM_HIDDEN_SIZE,
            num_layers   = cfg.LSTM_NUM_LAYERS,
            dropout      = cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0,
            batch_first  = True
        )

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class ODDecoder(nn.Module):
    """MLP decoder — same as GAT+LSTM."""

    def __init__(self, cfg: HypergraphConfig, scenario_feat_dim: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_SIZE + scenario_feat_dim, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, cfg.OUTPUT_SIZE)
        )

    def forward(self, lstm_out, scenario_feat):
        return self.decoder(torch.cat([lstm_out, scenario_feat], dim=-1))


class HypergraphLSTMModel(nn.Module):
    """
    Hypergraph+LSTM
    Pipeline: zone hypergraph (BKK lines) → HNN encoding → LSTM → ΔOD prediciton
    """

    def __init__(self, cfg: HypergraphConfig, scenario_feat_dim: int = 8):
        super().__init__()
        self.cfg          = cfg
        self.hg_encoder   = HypergraphEncoder(cfg)
        self.lstm_encoder = LSTMEncoder(cfg)
        self.od_decoder   = ODDecoder(cfg, scenario_feat_dim)
        self.H = self.D_v_inv_sqrt = self.D_e_inv = None

    def set_hypergraph(self, H: torch.Tensor):
        H_norm, D_v, D_e = normalize_incidence(H)
        self.H            = H_norm.to(self.cfg.DEVICE)
        self.D_v_inv_sqrt = D_v.to(self.cfg.DEVICE)
        self.D_e_inv      = D_e.to(self.cfg.DEVICE)
        print(f'✅ Hypergraph: {H.shape[0]} zone, {H.shape[1]} hyperedge')

    def forward(self, x_seq: List[torch.Tensor],
                scenario_feat: torch.Tensor) -> torch.Tensor:
        assert self.H is not None, 'Run before: model.set_hypergraph(H)'
        embeddings = [
            self.hg_encoder(x, self.H, self.D_v_inv_sqrt, self.D_e_inv).flatten()
            for x in x_seq
        ]
        seq      = torch.stack(embeddings, dim=0).unsqueeze(0)
        lstm_out = self.lstm_encoder(seq)
        return self.od_decoder(lstm_out, scenario_feat)


def train_epoch(model, optimizer, criterion, batch_data):
    model.train()
    optimizer.zero_grad()
    pred = model(batch_data['x_seq'], batch_data['scenario_feat'])
    loss = criterion(pred, batch_data['target'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, criterion, batch_data):
    model.eval()
    with torch.no_grad():
        pred = model(batch_data['x_seq'], batch_data['scenario_feat'])
        loss = criterion(pred, batch_data['target'])
        mae  = F.l1_loss(pred, batch_data['target'])
    return loss.item(), mae.item()


def train(model, cfg, train_data, val_data=None, save_path='hg_lstm_best.pt'):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    print(f'Training — device: {cfg.DEVICE} | parameters: {sum(p.numel() for p in model.parameters()):,}')

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_data)
        history['train_loss'].append(train_loss)

        if val_data is not None:
            val_loss, val_mae = evaluate(model, criterion, val_data)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAE: {val_mae:.4f} ✅')
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}')
            if patience_counter >= cfg.PATIENCE:
                print(f'Early stopping — {epoch+1} epoch')
                break
        else:
            if epoch % 10 == 0:
                print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f}')

    print(f'Done. Best val loss: {best_val_loss:.4f}')
    return history
