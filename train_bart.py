"""
train_bart.py — Train GAT+LSTM and Hypergraph+LSTM on BART data.

Usage (run from repo root):
    python train_bart.py --model gat        --epochs 300 --output checkpoints/bart_gat.pt
    python train_bart.py --model hypergraph --epochs 300 --output checkpoints/bart_hg.pt
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gat_lstm import GATLSTMModel, Config as GATConfig
from models.hypergraph_lstm import HypergraphLSTMModel, HypergraphConfig
from utils.bart_data import load_bart_transfer_dataset

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mae  = float(np.abs(pred - target).mean())
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Convert scenario to model inputs
# ---------------------------------------------------------------------------

def scenario_to_inputs(graph_data, scenario, device):
    x             = torch.tensor(scenario["node_features"], dtype=torch.float, device=device)
    x_seq         = [x]
    edge_index    = graph_data["edge_index"].to(device)
    # unsqueeze(0) → shape (1, 8) to match lstm_out shape (1, hidden)
    scenario_feat = torch.zeros(8, dtype=torch.float, device=device).unsqueeze(0)

    delta_od  = scenario["delta_od"]
    target_np = delta_od.sum(axis=1)
    std       = float(delta_od.std()) + 1e-8
    target    = torch.tensor(target_np / std, dtype=torch.float, device=device)

    return x_seq, edge_index, scenario_feat, target, std


# ---------------------------------------------------------------------------
# Build incidence matrix H for hypergraph model
# ---------------------------------------------------------------------------

def build_incidence_matrix(graph_data, n_nodes, device):
    he_index = graph_data["hyperedge_index"]
    n_edges  = graph_data["n_hyperedges"]

    if he_index.shape[1] == 0 or n_edges == 0:
        print("  [warning] Empty hyperedge_index — using identity hypergraph")
        return torch.eye(n_nodes, dtype=torch.float, device=device)

    H           = torch.zeros(n_nodes, n_edges, dtype=torch.float, device=device)
    station_idx = he_index[0].clamp(0, n_nodes - 1)
    edge_idx    = he_index[1].clamp(0, n_edges - 1)
    H[station_idx, edge_idx] = 1.0
    return H


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, model_type, optimizer,
                    train_scenarios, graph_data, device,
                    real_weight=5.0):
    model.train()
    total_loss = 0.0
    for idx in np.random.permutation(len(train_scenarios)):
        scenario = train_scenarios[int(idx)]
        x_seq, edge_index, sf, target, _ = scenario_to_inputs(
            graph_data, scenario, device
        )
        weight = real_weight if scenario["is_real"] else 1.0
        optimizer.zero_grad()

        if model_type == "gat":
            pred = model(x_seq, edge_index, sf)
        else:
            pred = model(x_seq, sf)

        loss = weight * F.mse_loss(pred.squeeze(), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(train_scenarios), 1)


@torch.no_grad()
def evaluate(model, model_type, val_scenarios, graph_data, device):
    model.eval()
    all_preds, all_targets = [], []
    for scenario in val_scenarios:
        x_seq, edge_index, sf, _, std = scenario_to_inputs(
            graph_data, scenario, device
        )
        if model_type == "gat":
            pred = model(x_seq, edge_index, sf)
        else:
            pred = model(x_seq, sf)

        all_preds.append(pred.squeeze().cpu().numpy() * std)
        all_targets.append(scenario["delta_od"].sum(axis=1))

    return compute_metrics(
        np.concatenate(all_preds),
        np.concatenate(all_targets)
    )


def extract_encoder_state(model):
    prefixes = ("gat_encoder", "hg_encoder", "lstm_encoder")
    return {k: v.cpu() for k, v in model.state_dict().items()
            if any(k.startswith(p) for p in prefixes)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["gat", "hypergraph"], default="gat")
    parser.add_argument("--epochs",      type=int,   default=300)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--output",      type=str,   default="checkpoints/bart_pretrained.pt")
    parser.add_argument("--data_dir",    type=str,   default="data/bart")
    parser.add_argument("--n_synthetic", type=int,   default=60)
    parser.add_argument("--patience",    type=int,   default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    graph_data, scenarios = load_bart_transfer_dataset(
        data_dir=args.data_dir, n_synthetic=args.n_synthetic, verbose=True,
    )

    train_scenarios = [s for s in scenarios if s["split"] == "train"]
    val_scenarios   = [s for s in scenarios if s["split"] == "val"]
    n_nodes         = graph_data["n_nodes"]

    print(f"Nodes: {n_nodes} | Train: {len(train_scenarios)} | Val: {len(val_scenarios)}")
 # ── Build configs dynamically from n_nodes ────────────────────────────
    class BARTGATConfig(GATConfig):
        NUM_ZONES       = n_nodes
        GAT_IN_CHANNELS = 16
        OUTPUT_SIZE     = n_nodes
        DEVICE          = str(device)

    class BARTHGConfig(HypergraphConfig):
        NUM_ZONES      = n_nodes
        HG_IN_CHANNELS = 16
        OUTPUT_SIZE    = n_nodes
        DEVICE         = str(device)
    # ─────────────────────────────────────────────────────────────────────
    H = None
    if args.model == "gat":
        model = GATLSTMModel(cfg=BARTGATConfig, scenario_feat_dim=8).to(device)
    else:
        model = HypergraphLSTMModel(cfg=BARTHGConfig, scenario_feat_dim=8).to(device)
        H = build_incidence_matrix(graph_data, n_nodes, device)
        model.set_hypergraph(H)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}+LSTM | Params: {n_params:,}")

    # Forward test
    model.eval()
    with torch.no_grad():
        x_seq, edge_index, sf, tgt, _ = scenario_to_inputs(
            graph_data, train_scenarios[0], device
        )
        out = model(x_seq, edge_index, sf) if args.model == "gat" \
              else model(x_seq, sf)
    print(f"Forward OK — output: {tuple(out.shape)}, target: {tuple(tgt.shape)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    best_mae, best_r2, best_epoch, patience_ctr = float("inf"), -float("inf"), 0, 0
    print(f"\nTraining {args.epochs} epochs (patience={args.patience})...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, args.model, optimizer, train_scenarios, graph_data, device
        )
        scheduler.step()

        if val_scenarios:
            m        = evaluate(model, args.model, val_scenarios, graph_data, device)
            improved = m["mae"] < best_mae
            if improved:
                best_mae, best_r2, best_epoch, patience_ctr = \
                    m["mae"], m["r2"], epoch, 0
                torch.save({
                    "encoder_state": extract_encoder_state(model),
                    "full_state":    model.state_dict(),
                    "model_type":    args.model,
                    "val_mae":       best_mae,
                    "val_r2":        best_r2,
                    "epoch":         epoch,
                    "n_nodes":       n_nodes,
                }, args.output)
            else:
                patience_ctr += 1

            if epoch % 10 == 0 or epoch <= 5:
                print(
                    f"Ep {epoch:4d}/{args.epochs} | loss={loss:.4f} | "
                    f"val_mae={m['mae']:.2f} | val_r2={m['r2']:+.4f} | "
                    f"best={best_mae:.2f}@{best_epoch} | "
                    f"{'✓' if improved else ' '} | "
                    f"{(time.time()-t0)/60:.1f}min"
                )

            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        else:
            if epoch % 20 == 0:
                print(f"Ep {epoch:4d}/{args.epochs} | loss={loss:.4f}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Best: ep{best_epoch} | mae={best_mae:.2f} | r2={best_r2:+.4f}")
    print(f"Saved: {args.output}")

    if val_scenarios and os.path.exists(args.output):
        print("\n── Final Berryessa validation ──")
        ckpt = torch.load(args.output, map_location=device)
        model.load_state_dict(ckpt["full_state"])
        if args.model == "hypergraph" and H is not None:
            model.set_hypergraph(H)
        f = evaluate(model, args.model, val_scenarios, graph_data, device)
        print(f"  MAE:  {f['mae']:.3f} passengers/zone")
        print(f"  RMSE: {f['rmse']:.3f}")
        print(f"  R²:   {f['r2']:+.4f}")


if __name__ == "__main__":
    main()