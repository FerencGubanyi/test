"""
train_bart.py — Train both architectures on BART data for transfer learning.

Run this on Saturday afternoon on Colab Pro (T4/A100):

    python train_bart.py --model gat --epochs 300 --output checkpoints/bart_gat.pt
    python train_bart.py --model hypergraph --epochs 300 --output checkpoints/bart_hg.pt

The saved checkpoints contain only the encoder weights (GAT/LSTM layers).
The prediction head is NOT saved — it is city-specific and will be randomly
initialised during BKK fine-tuning.

Checkpoint format (compatible with transfer.py):
    {
        'encoder_state': OrderedDict,   # weights of embedding+GAT+LSTM
        'config': {...},                # hyperparameters
        'val_mae': float,               # best validation MAE on Berryessa scenario
        'val_r2':  float,
        'epoch':   int,
    }
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from utils.bart_data import load_bart_transfer_dataset

# ── Import your existing model classes ──────────────────────────────────────
# Adjust these imports to match the actual module paths in your repo.
try:
    from models.gat_lstm import GATLSTMModel
    from models.hypergraph_lstm import HypergraphLSTMModel
except ImportError as e:
    raise ImportError(
        f"Could not import model classes: {e}\n"
        "Make sure you run this script from the repo root directory."
    )


# ---------------------------------------------------------------------------
# Metrics (same as BKK pipeline)
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    mae  = float(np.abs(pred - target).mean())
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ---------------------------------------------------------------------------
# Loss with real-scenario weighting (same logic as BKK train.py)
# ---------------------------------------------------------------------------

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    return weight * F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# Build PyG Data object from BART graph + scenario
# ---------------------------------------------------------------------------

def scenario_to_pyg(
    graph_data: dict,
    scenario: dict,
    device: torch.device,
) -> tuple[Data, torch.Tensor]:
    """
    Convert a BART scenario dict into a PyG Data object + target tensor.
    Mirrors the BKK data loader behaviour.
    """
    node_features = torch.tensor(
        scenario["node_features"], dtype=torch.float, device=device
    )
    edge_index = graph_data["edge_index"].to(device)
    edge_attr  = graph_data["edge_attr"].to(device)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    # Hyperedge incidence for HypergraphLSTMModel
    data.hyperedge_index = graph_data["hyperedge_index"].to(device)
    data.num_hyperedges  = graph_data["n_hyperedges"]

    target = torch.tensor(
        scenario["delta_od_normalized"].flatten(), dtype=torch.float, device=device
    )

    return data, target


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_scenarios: list[dict],
    graph_data: dict,
    device: torch.device,
    real_weight: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    # Shuffle training order
    indices = np.random.permutation(len(train_scenarios))

    for idx in indices:
        scenario = train_scenarios[idx]
        data, target = scenario_to_pyg(graph_data, scenario, device)
        weight = real_weight if scenario["is_real"] else 1.0

        optimizer.zero_grad()
        pred = model(data)
        loss = weighted_mse_loss(pred, target, weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_scenarios: list[dict],
    graph_data: dict,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds, all_targets = [], []

    for scenario in val_scenarios:
        data, target = scenario_to_pyg(graph_data, scenario, device)
        pred = model(data)

        # Denormalise
        std = scenario["std"]
        pred_denorm   = pred.cpu().numpy() * std
        target_denorm = scenario["delta_od"].flatten()

        all_preds.append(pred_denorm)
        all_targets.append(target_denorm)

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    return compute_metrics(preds, targets)


# ---------------------------------------------------------------------------
# Encoder extraction utility
# ---------------------------------------------------------------------------

def extract_encoder_state(model: nn.Module) -> dict:
    """
    Extract only the encoder layers (embedding + GAT/hypergraph conv + LSTM).
    Excludes the prediction head (city-specific).
    Returns an OrderedDict ready for load_state_dict in transfer.py.
    """
    encoder_prefixes = (
        "embedding", "gat1", "gat2",          # GAT+LSTM
        "hgnn", "hgnn1", "hgnn2",             # HypergraphLSTM (adjust to match your class)
        "lstm", "layer_norm",
    )
    state = {}
    for name, param in model.state_dict().items():
        if any(name.startswith(p) for p in encoder_prefixes):
            state[name] = param.cpu()
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train on BART for transfer learning")
    parser.add_argument("--model",   choices=["gat", "hypergraph"], default="gat")
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--output",  type=str, default="checkpoints/bart_pretrained.pt")
    parser.add_argument("--data_dir",type=str, default="data/bart")
    parser.add_argument("--n_synthetic", type=int, default=60)
    parser.add_argument("--patience",   type=int, default=40,
                        help="Early stopping patience (epochs)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load BART data ──────────────────────────────────────────────────────
    graph_data, scenarios = load_bart_transfer_dataset(
        data_dir=args.data_dir,
        n_synthetic=args.n_synthetic,
        verbose=True,
    )

    train_scenarios = [s for s in scenarios if s["split"] == "train"]
    val_scenarios   = [s for s in scenarios if s["split"] == "val"]

    print(f"Train scenarios: {len(train_scenarios)}")
    print(f"Val scenarios:   {len(val_scenarios)}")

    n_nodes    = graph_data["n_nodes"]
    n_features = train_scenarios[0]["node_features"].shape[1]  # 16

    # ── Build model ──────────────────────────────────────────────────────────
    # Output size: n_nodes * n_nodes (flattened ΔOD matrix)
    output_size = n_nodes * n_nodes

    model_config = {
        "n_zones":     n_nodes,
        "n_features":  n_features,
        "hidden_dim":  64,
        "n_heads":     8,
        "lstm_layers": 2,
        "dropout":     0.1,
        "output_size": output_size,
    }

    if args.model == "gat":
        model = GATLSTMModel(**model_config).to(device)
    else:
        model = HypergraphLSTMModel(
            **model_config,
            n_hyperedges=graph_data["n_hyperedges"],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()}+LSTM  |  Parameters: {n_params:,}")

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    best_val_mae = float("inf")
    best_val_r2  = -float("inf")
    best_epoch   = 0
    patience_counter = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_scenarios, graph_data, device
        )
        scheduler.step()

        if val_scenarios:
            val_metrics = evaluate(model, val_scenarios, graph_data, device)
            val_mae = val_metrics["mae"]
            val_r2  = val_metrics["r2"]

            improved = val_mae < best_val_mae
            if improved:
                best_val_mae = val_mae
                best_val_r2  = val_r2
                best_epoch   = epoch
                patience_counter = 0

                # Save checkpoint
                torch.save({
                    "encoder_state": extract_encoder_state(model),
                    "full_state":    model.state_dict(),
                    "config":        model_config,
                    "model_type":    args.model,
                    "val_mae":       best_val_mae,
                    "val_r2":        best_val_r2,
                    "epoch":         epoch,
                }, args.output)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch <= 5:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:4d}/{args.epochs} | "
                    f"loss={train_loss:.4f} | "
                    f"val_mae={val_mae:.3f} | "
                    f"val_r2={val_r2:.4f} | "
                    f"best={best_val_mae:.3f}@ep{best_epoch} | "
                    f"{'✓' if improved else ' '} | "
                    f"{elapsed/60:.1f}min"
                )

            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break
        else:
            # No val set — just log training loss
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d}/{args.epochs} | loss={train_loss:.4f}")
            # Save periodically
            if epoch % 50 == 0:
                torch.save({
                    "encoder_state": extract_encoder_state(model),
                    "full_state":    model.state_dict(),
                    "config":        model_config,
                    "model_type":    args.model,
                    "val_mae":       None,
                    "val_r2":        None,
                    "epoch":         epoch,
                }, args.output)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Best checkpoint: epoch {best_epoch} | "
          f"val_mae={best_val_mae:.3f} | val_r2={best_val_r2:.4f}")
    print(f"Saved to: {args.output}")

    # Final evaluation on val set
    if val_scenarios:
        print("\n── Final evaluation on Berryessa validation scenario ──")
        # Load best checkpoint
        ckpt = torch.load(args.output, map_location=device)
        model.load_state_dict(ckpt["full_state"])
        final_metrics = evaluate(model, val_scenarios, graph_data, device)
        print(f"  MAE:  {final_metrics['mae']:.3f} passengers/station-pair")
        print(f"  RMSE: {final_metrics['rmse']:.3f}")
        print(f"  R²:   {final_metrics['r2']:.4f}")
        print()
        print("The checkpoint is ready for transfer.py → BKK fine-tuning.")


if __name__ == "__main__":
    main()
