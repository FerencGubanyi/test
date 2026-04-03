"""
GAT+LSTM Model

The model stands on 2 component:
- GAT (Graph Attention Network): the analysis between the connected zones
- LSTM: the timeline connection between zones
- Output: ΔOD matrix (1419 zone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os


class Config:
    # Static datas
    NUM_ZONES       = 1419
    NUM_SCENARIOS   = 3

    # GAT
    GAT_IN_CHANNELS  = 16
    GAT_HIDDEN       = 64
    GAT_OUT_CHANNELS = 32
    GAT_HEADS        = 4
    GAT_DROPOUT      = 0.2

    # LSTM
    LSTM_INPUT_SIZE  = 32
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS  = 2
    LSTM_DROPOUT     = 0.2

    OUTPUT_SIZE = NUM_ZONES

    BATCH_SIZE    = 4
    LEARNING_RATE = 1e-3
    NUM_EPOCHS    = 100
    PATIENCE      = 15

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GATEncoder(nn.Module):
    """
    GAT encoder: it is learns a vector for every neighbour zones,
    aggergate it with attentions weights
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.gat1 = GATConv(
            in_channels  = cfg.GAT_IN_CHANNELS,
            out_channels = cfg.GAT_HIDDEN,
            heads        = cfg.GAT_HEADS,
            dropout      = cfg.GAT_DROPOUT,
            concat       = True
        )
        self.gat2 = GATConv(
            in_channels  = cfg.GAT_HIDDEN * cfg.GAT_HEADS,
            out_channels = cfg.GAT_OUT_CHANNELS,
            heads        = 1,
            dropout      = cfg.GAT_DROPOUT,
            concat       = False
        )
        self.dropout = nn.Dropout(cfg.GAT_DROPOUT)
        self.bn1 = nn.BatchNorm1d(cfg.GAT_HIDDEN * cfg.GAT_HEADS)
        self.bn2 = nn.BatchNorm1d(cfg.GAT_OUT_CHANNELS)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        return x


class LSTMEncoder(nn.Module):
    """
    LSTM encoder:learns temporal patterns from a sequence of zone embeddings
    produced by GAT
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.input_proj = nn.Linear(
            cfg.NUM_ZONES * cfg.GAT_OUT_CHANNELS,
            cfg.LSTM_INPUT_SIZE * 8
        )
        self.lstm = nn.LSTM(
            input_size   = cfg.LSTM_INPUT_SIZE * 8,
            hidden_size  = cfg.LSTM_HIDDEN_SIZE,
            num_layers   = cfg.LSTM_NUM_LAYERS,
            dropout      = cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0,
            batch_first  = True
        )
        self.dropout = nn.Dropout(cfg.LSTM_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.relu(x)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class ODDecoder(nn.Module):
    """
    MLP decoder: from the LSTM output and the scenario description
    it produces the predicted ΔOD vector per zone.
    """

    def __init__(self, cfg: Config, scenario_feat_dim: int = 8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_SIZE + scenario_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, cfg.OUTPUT_SIZE)
        )

    def forward(self, lstm_out: torch.Tensor,
                scenario_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([lstm_out, scenario_feat], dim=-1)
        return self.decoder(x)


class GATLSTMModel(nn.Module):
    """
    Complete GAT+LSTM architecture.
    Pipeline: zone graph → GAT encoding → LSTM → ΔOD prediction
    """

    def __init__(self, cfg: Config, scenario_feat_dim: int = 8):
        super().__init__()
        self.gat_encoder  = GATEncoder(cfg)
        self.lstm_encoder = LSTMEncoder(cfg)
        self.od_decoder   = ODDecoder(cfg, scenario_feat_dim)

    def forward(self, x_seq: List[torch.Tensor],
                edge_index: torch.Tensor,
                scenario_feat: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for x in x_seq:
            emb = self.gat_encoder(x, edge_index)
            embeddings.append(emb.flatten())
        seq      = torch.stack(embeddings, dim=0).unsqueeze(0)
        lstm_out = self.lstm_encoder(seq)
        return self.od_decoder(lstm_out, scenario_feat)


def build_zone_graph(zone_ids: List[int], gdf=None,
                     k_neighbors: int = 6) -> torch.Tensor:
    """Build up neighbouring graph."""
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
                return torch.tensor(edges, dtype=torch.long).t()
        except Exception as e:
            print(f'The graph building went wrong: {e}, k-NN fallback')

    edges = []
    for i in range(n):
        neighbors = np.random.choice(
            [j for j in range(n) if j != i],
            size=min(k_neighbors, n-1), replace=False
        )
        for j in neighbors:
            edges.append([i, j])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def od_matrix_to_zone_features(od_matrix: pd.DataFrame,
                                cfg: Config) -> torch.Tensor:
    """Create 16 dimension zone feature vector from OD matrix"""
    features = []
    for zone_id in od_matrix.index:
        row = od_matrix.loc[zone_id].values.astype(float)
        col = (od_matrix[zone_id].values.astype(float)
               if zone_id in od_matrix.columns
               else np.zeros(len(od_matrix)))
        feat = [
            row.sum(), col.sum(),
            row.mean(), row.std(),
            col.mean(), col.std(),
            (row > 0).sum(), (col > 0).sum(),
            row.max(), col.max(),
            np.percentile(row, 75), np.percentile(col, 75),
            np.percentile(row, 25), np.percentile(col, 25),
            row.sum() / (col.sum() + 1e-6),
            np.log1p(row.sum()),
        ]
        features.append(feat)
    feat_tensor = torch.tensor(features, dtype=torch.float32)
    feat_tensor = (feat_tensor - feat_tensor.mean(dim=0)) / (feat_tensor.std(dim=0) + 1e-8)
    assert feat_tensor.shape == (len(od_matrix), cfg.GAT_IN_CHANNELS)
    return feat_tensor


def build_scenario_features(scenario_type: str, affected_zones: List[int],
                              num_new_stops: int = 0) -> torch.Tensor:
    """Infrastrucue changes store in feature vectors"""
    enc = {'metro_extension': [1,0,0], 'bus_new': [0,1,0], 'tram_new': [0,0,1]}
    feat = enc.get(scenario_type, [0,0,0]) + [
        len(affected_zones), num_new_stops,
        np.log1p(len(affected_zones)),
        min(len(affected_zones) / 100, 1.0),
        1.0 if scenario_type == 'metro_extension' else 0.0,
    ]
    return torch.tensor([feat], dtype=torch.float32)


def train_epoch(model, optimizer, criterion, batch_data, cfg):
    model.train()
    optimizer.zero_grad()
    pred = model(batch_data['x_seq'], batch_data['edge_index'], batch_data['scenario_feat'])
    loss = criterion(pred, batch_data['target'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, criterion, batch_data):
    model.eval()
    with torch.no_grad():
        pred = model(batch_data['x_seq'], batch_data['edge_index'], batch_data['scenario_feat'])
        loss = criterion(pred, batch_data['target'])
        mae  = F.l1_loss(pred, batch_data['target'])
    return loss.item(), mae.item()


def train(model, cfg, train_data, val_data=None, save_path='gat_lstm_best.pt'):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [], 'val_mae': []}

    print(f'Training — device: {cfg.DEVICE} | parameters: {sum(p.numel() for p in model.parameters()):,}')

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
