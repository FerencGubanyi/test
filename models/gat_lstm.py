"""
models/gat_lstm.py
GAT+LSTM model for zone-level OD redistribution prediction.

Pipeline: zone graph → GAT spatial encoding → LSTM → ΔOD vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
from typing import List, Optional


class Config:
    NUM_ZONES        = 1419

    # GAT
    GAT_IN_CHANNELS  = 22    # 16 OD-based + 6 GTFS features
    GAT_HIDDEN       = 64
    GAT_OUT_CHANNELS = 32
    GAT_HEADS        = 4
    GAT_DROPOUT      = 0.2

    # LSTM
    LSTM_INPUT_SIZE  = 32
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS  = 2
    LSTM_DROPOUT     = 0.2

    OUTPUT_SIZE   = NUM_ZONES
    LEARNING_RATE = 1e-3
    NUM_EPOCHS    = 100
    PATIENCE      = 15

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GATEncoder(nn.Module):
    """Two-layer GAT with batch norm and residual connection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gat1 = GATConv(
            cfg.GAT_IN_CHANNELS,
            cfg.GAT_HIDDEN,
            heads=cfg.GAT_HEADS,
            dropout=cfg.GAT_DROPOUT,
            concat=True,
        )
        self.gat2 = GATConv(
            cfg.GAT_HIDDEN * cfg.GAT_HEADS,
            cfg.GAT_OUT_CHANNELS,
            heads=1,
            dropout=cfg.GAT_DROPOUT,
            concat=False,
        )
        self.bn1     = nn.BatchNorm1d(cfg.GAT_HIDDEN * cfg.GAT_HEADS)
        self.bn2     = nn.BatchNorm1d(cfg.GAT_OUT_CHANNELS)
        self.dropout = nn.Dropout(cfg.GAT_DROPOUT)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.bn1(self.gat1(x, edge_index)))
        x = self.dropout(x)
        x = F.elu(self.bn2(self.gat2(x, edge_index)))
        return x


class LSTMEncoder(nn.Module):
    """Projects flattened GAT embeddings and encodes the scenario sequence."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(
            cfg.NUM_ZONES * cfg.GAT_OUT_CHANNELS,
            cfg.LSTM_INPUT_SIZE * 8,
        )
        self.lstm = nn.LSTM(
            input_size=cfg.LSTM_INPUT_SIZE * 8,
            hidden_size=cfg.LSTM_HIDDEN_SIZE,
            num_layers=cfg.LSTM_NUM_LAYERS,
            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class ODDecoder(nn.Module):
    """MLP that maps LSTM output + scenario features to a ΔOD vector."""

    def __init__(self, cfg: Config, scenario_feat_dim: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_SIZE + scenario_feat_dim, 512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, cfg.OUTPUT_SIZE),
        )

    def forward(self, lstm_out: torch.Tensor,
                scenario_feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([lstm_out, scenario_feat], dim=-1))


class GATLSTMModel(nn.Module):
    """
    Full GAT+LSTM architecture.
    Forward: list of zone feature tensors (one per scenario step) →
             edge_index → scenario_feat → ΔOD prediction (1 x N_zones)
    """

    def __init__(self, cfg: Config, scenario_feat_dim: int = 8):
        super().__init__()
        self.gat_encoder  = GATEncoder(cfg)
        self.lstm_encoder = LSTMEncoder(cfg)
        self.od_decoder   = ODDecoder(cfg, scenario_feat_dim)

    def forward(self, x_seq: List[torch.Tensor],
                edge_index: torch.Tensor,
                scenario_feat: torch.Tensor) -> torch.Tensor:
        embeddings = [self.gat_encoder(x, edge_index).flatten() for x in x_seq]
        seq      = torch.stack(embeddings, dim=0).unsqueeze(0)
        lstm_out = self.lstm_encoder(seq)
        return self.od_decoder(lstm_out, scenario_feat)


#      Graph construction                

def build_zone_graph(zone_ids: List[int], gdf=None,
                     k_neighbors: int = 6) -> torch.Tensor:
    """
    Build a zone adjacency graph.
    Uses shapefile topology (touches) when available,
    falls back to random k-NN otherwise.
    """
    n = len(zone_ids)

    if gdf is not None:
        try:
            edges = []
            for i, row_i in gdf.iterrows():
                for j, row_j in gdf.iterrows():
                    if i != j and row_i.geometry.touches(row_j.geometry):
                        zi = zone_ids.index(int(row_i['NO']))
                        zj = zone_ids.index(int(row_j['NO']))
                        edges.append([zi, zj])
            if edges:
                return torch.tensor(edges, dtype=torch.long).t().contiguous()
        except Exception as e:
            print(f'Graph build failed ({e}), using k-NN fallback')

    rng = np.random.default_rng(42)
    edges = []
    for i in range(n):
        neighbors = rng.choice(
            [j for j in range(n) if j != i],
            size=min(k_neighbors, n - 1), replace=False,
        )
        for j in neighbors:
            edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


#      Training utilities                

def train_epoch(model, optimizer, criterion, batch_data, cfg):
    model.train()
    optimizer.zero_grad()
    pred = model(batch_data['x_seq'], batch_data['edge_index'],
                 batch_data['scenario_feat'])
    loss = criterion(pred, batch_data['target'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, criterion, batch_data):
    model.eval()
    with torch.no_grad():
        pred = model(batch_data['x_seq'], batch_data['edge_index'],
                     batch_data['scenario_feat'])
        loss = criterion(pred, batch_data['target'])
        mae  = F.l1_loss(pred, batch_data['target'])
    return loss.item(), mae.item()


def train(model, cfg, train_data, val_data=None,
          save_path='gat_lstm_best.pt'):
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    print(f'Training — device: {cfg.DEVICE} | '
          f'params: {sum(p.numel() for p in model.parameters()):,}')

    for epoch in range(cfg.NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, criterion, train_data, cfg)
        history['train_loss'].append(train_loss)

        if val_data is not None:
            val_loss, val_mae = evaluate(model, criterion, val_data)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'best_epoch':       epoch + 1,
                    'best_val_loss':    best_val_loss,
                    'train_losses':     history['train_loss'],
                    'val_losses':       history['val_loss'],
                    'val_maes':         history['val_mae'],
                }, save_path)
                print(f'  Epoch {epoch+1:3d} | '
                      f'Train: {train_loss:.4f} | '
                      f'Val: {val_loss:.4f} | '
                      f'MAE: {val_mae:.4f} ✅')
            else:
                patience_counter += 1
                if epoch % 10 == 0:
                    print(f'  Epoch {epoch+1:3d} | '
                          f'Train: {train_loss:.4f} | '
                          f'Val: {val_loss:.4f}')

            if patience_counter >= cfg.PATIENCE:
                print(f'Early stopping at epoch {epoch + 1}')
                break
        else:
            if epoch % 10 == 0:
                print(f'  Epoch {epoch+1:3d} | Train: {train_loss:.4f}')

    print(f'Done. Best val loss: {best_val_loss:.4f}')
    return history