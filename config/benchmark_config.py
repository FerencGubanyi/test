"""
benchmark_metr_la.py  (v2 — real model interface)
--------------------------------------------------

METR-LA adaptation:
  N = 207 sensors (vs 1419 BKK zones).
  x_seq[0] = first half of T_in window  -> "base state"   (N, T_in//2)
  x_seq[1] = second half of T_in window -> "current state" (N, T_in//2)
  target   = mean speed per node over next T_out steps, shape (1, N)
  scenario_feat = zeros (1, 8)  — no topology change in METR-LA

Run on Colab:
    !python benchmark_metr_la.py --model gat        --epochs 50
    !python benchmark_metr_la.py --model hypergraph --epochs 50
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils.metr_la_loader import get_metr_dataloaders


# ---------------------------------------------------------------------------
# Config patching helpers
# ---------------------------------------------------------------------------

def _make_gat_config(n_zones, in_channels, hidden, heads):
    from models.gat_lstm import Config
    cfg = Config()
    cfg.NUM_ZONES       = n_zones
    cfg.OUTPUT_SIZE     = n_zones
    cfg.GAT_IN_CHANNELS = in_channels
    cfg.GAT_HIDDEN      = hidden
    cfg.GAT_HEADS       = heads
    return cfg


def _make_hg_config(n_zones, in_channels, hidden):
    from models.hypergraph_lstm import HypergraphConfig
    cfg = HypergraphConfig()
    cfg.NUM_ZONES      = n_zones
    cfg.OUTPUT_SIZE    = n_zones
    cfg.HG_IN_CHANNELS = in_channels
    cfg.HG_HIDDEN      = hidden
    cfg.LSTM_INPUT_SIZE = hidden
    return cfg


# ---------------------------------------------------------------------------
# Batch → x_seq conversion
# ---------------------------------------------------------------------------

def batch_to_x_seq(node_feat: torch.Tensor, device):
    """
    node_feat : (B, N, T_in)
    Returns   : [tensor(N, T_in//2), tensor(N, T_in//2)]  — first sample only.

    The models flatten ALL node embeddings into a single LSTM vector, so
    true batching across graph samples doesn't apply here. We process one
    sample per forward pass (matches the per-scenario BKK training loop).
    """
    x  = node_feat[0]             # (N, T_in)
    h  = x.shape[1] // 2
    return [x[:, :h].to(device), x[:, h:].to(device)]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r2(pred, target):
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()

def mae_mph(pred, target, mean, std):
    return ((pred - target).abs() * std).mean().item()

def rmse_mph(pred, target, mean, std):
    return (((pred - target) * std) ** 2).mean().sqrt().item()


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_epoch(model, mtype, loader, optimizer, edge_index, device, sf):
    model.train()
    losses = []
    for batch in loader:
        x_seq = batch_to_x_seq(batch["node_feat"], device)
        y     = batch["target"][0].unsqueeze(0).to(device)  # (1, N)

        optimizer.zero_grad()
        pred = model(x_seq, edge_index, sf.to(device)) if mtype == "gat" \
               else model(x_seq, sf.to(device))
        loss = F.mse_loss(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_epoch(model, mtype, loader, edge_index, device, sf, speed_mean, speed_std):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        x_seq = batch_to_x_seq(batch["node_feat"], device)
        y     = batch["target"][0].unsqueeze(0).to(device)

        pred = model(x_seq, edge_index, sf.to(device)) if mtype == "gat" \
               else model(x_seq, sf.to(device))
        preds.append(pred.squeeze(0).cpu())
        tgts.append(y.squeeze(0).cpu())

    p = torch.stack(preds).view(-1)
    t = torch.stack(tgts).view(-1)
    return {
        "MAE":  mae_mph(p, t, speed_mean, speed_std),
        "RMSE": rmse_mph(p, t, speed_mean, speed_std),
        "R2":   r2(p, t),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="METR-LA benchmark (real models)")
    parser.add_argument("--model",      default="gat", choices=["gat", "hypergraph"])
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden",     type=int,   default=64)
    parser.add_argument("--gat_heads",  type=int,   default=4)
    parser.add_argument("--T_in",       type=int,   default=12)
    parser.add_argument("--T_out",      type=int,   default=12)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--patience",   type=int,   default=10)
    parser.add_argument("--cache_dir",  default=None)
    parser.add_argument("--save_path",  default="checkpoints/metr_la_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Model: {args.model}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = get_metr_dataloaders(
        cache_dir=args.cache_dir,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
    )
    N           = meta["num_nodes"]     # 207
    speed_mean  = meta["speed_mean"]
    speed_std   = meta["speed_std"]
    edge_index  = meta["edge_index"].to(device)
    feat_step   = args.T_in // 2        # feature dim per x_seq step

    # Dummy scenario feat — shape matches ODDecoder input (1, 8)
    scenario_feat = torch.zeros(1, 8)

    # ── Build model ──────────────────────────────────────────────────────────
    if args.model == "gat":
        cfg   = _make_gat_config(N, feat_step, args.hidden, args.gat_heads)
        from models.gat_lstm import GATLSTMModel
        model = GATLSTMModel(cfg).to(device)
        mtype = "gat"
        print(f"GAT_OUT_CHANNELS={cfg.GAT_OUT_CHANNELS}  "
              f"LSTM_proj_in={N * cfg.GAT_OUT_CHANNELS}")

    else:
        cfg   = _make_hg_config(N, feat_step, args.hidden)
        from models.hypergraph_lstm import HypergraphLSTMModel, build_incidence_matrix
        model = HypergraphLSTMModel(cfg).to(device)
        mtype = "hypergraph"

        # Derive route-like hyperedges from the sensor adjacency graph:
        # each sensor's 1-hop neighbourhood becomes one hyperedge.
        src, dst = edge_index.cpu()
        route_dict = defaultdict(list)
        for s, d in zip(src.tolist(), dst.tolist()):
            route_dict[s].append(d)
            route_dict[d].append(s)
        gtfs_routes = {f"route_{k}": list(set(v + [k]))
                       for k, v in route_dict.items()}
        H = build_incidence_matrix(list(range(N)), gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)
        print(f"HG hyperedges={H.shape[1]}  "
              f"LSTM_proj_in={N * cfg.HG_OUT_CHANNELS}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,}  |  N={N}  |  feat/step={feat_step}\n")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mae = float("inf")
    patience_ctr = 0
    save_path    = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'Ep':>4} | {'Train MSE':>10} | {'Val MAE':>8} | "
          f"{'Val RMSE':>9} | {'Val R²':>7} | {'s':>4}")
    print("-" * 58)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            model, mtype, train_loader, optimizer,
            edge_index, device, scenario_feat
        )
        vm = eval_epoch(
            model, mtype, val_loader,
            edge_index, device, scenario_feat, speed_mean, speed_std
        )
        scheduler.step()
        elapsed = time.time() - t0

        tag = ""
        if vm["MAE"] < best_val_mae:
            best_val_mae = vm["MAE"]
            patience_ctr = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val": vm, "args": vars(args),
                        "meta": meta}, save_path)
            tag = " ✅"
        else:
            patience_ctr += 1

        print(f"{epoch:>4} | {tr_loss:>10.5f} | {vm['MAE']:>8.4f} | "
              f"{vm['RMSE']:>9.4f} | {vm['R2']:>7.4f} | {elapsed:>3.1f}s{tag}")

        if patience_ctr >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    # ── Test ──────────────────────────────────────────────────────────────────
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tm = eval_epoch(model, mtype, test_loader,
                    edge_index, device, scenario_feat, speed_mean, speed_std)

    print("\n" + "=" * 58)
    print(f"TEST — {args.model.upper()} | METR-LA | T_in={args.T_in}")
    print("=" * 58)
    print(f"  MAE  : {tm['MAE']:.4f} mph")
    print(f"  RMSE : {tm['RMSE']:.4f} mph")
    print(f"  R²   : {tm['R2']:.4f}")
    print()
    print("Literature (MAE mph):  DCRNN 2.77 | STGCN 2.96 | ST-GAT 2.68")
    print("=" * 58)

    r = tm["R2"]
    if r > 0.85:
        print("\n✅  Architecture healthy. BKK alacsony R² = adathiány, nem modell.")
    elif r > 0.55:
        print("\n~   Közepes. Próbálj több epochot vagy nagyobb hidden dimet.")
        print("    Ha METR-LA R² > 0.55 és BKK ~ 0 → adathiány a probléma.")
    else:
        print("\n✗   Alacsony R² METR-LA-n is → architektúra-szintű probléma.")
        print("    Ellenőrizd: LSTM input proj dimenzió, gradient flow, LR.")

    return tm


if __name__ == "__main__":
    main()