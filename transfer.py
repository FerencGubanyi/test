"""
transfer.py — Transfer learning from BART/Shanghai → BKK fine-tuning.

Loads encoder weights from pretrained BART and/or Shanghai checkpoints,
freezes them, trains only the BKK prediction head, then progressively
unfreezes for full fine-tuning.

Usage:
    python transfer.py --model gat        --source bart
    python transfer.py --model hypergraph --source bart
    python transfer.py --model gat        --source shanghai
    python transfer.py --model gat        --source both

Phases:
    Phase 1 (epochs 1-50):   Frozen encoder, train head only
    Phase 2 (epochs 51-100): Unfreeze LSTM
    Phase 3 (epochs 101-200): Unfreeze everything (low LR)
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.paths import (
    BASE_DIR, ZONES_SHP, GTFS_ZIP, SYNTHETIC_DIR,
    M2_BASE_KK, M2_DEV_KK,
    M1_KK, M1_DIFF_KK,
    BUS35_KK, BUS35_DIFF_KK,
    GAT_CHECKPOINT, HG_CHECKPOINT,
)
from utils.data import (
    load_od_matrix_with_header,
    od_matrix_to_zone_features, build_gtfs_zone_features,
    diff_to_target, build_scenario_features, get_affected_zones,
    NUM_FEATURES,
)
from utils.synthetic_scenarios import load_scenarios
from utils.loss import combined_loss
from utils.metrics import evaluate_all


# ---------------------------------------------------------------------------
# Checkpoint paths for pretrained encoders
# ---------------------------------------------------------------------------

BART_GAT_CKPT      = "checkpoints/bart_gat.pt"
BART_HG_CKPT       = "checkpoints/bart_hg.pt"
SHANGHAI_GAT_CKPT  = "checkpoints/shanghai_gat.pt"
SHANGHAI_HG_CKPT   = "checkpoints/shanghai_hg.pt"

TRANSFER_GAT_CKPT  = "checkpoints/transfer_gat.pt"
TRANSFER_HG_CKPT   = "checkpoints/transfer_hg.pt"


# ---------------------------------------------------------------------------
# Encoder weight loading
# ---------------------------------------------------------------------------

def load_encoder_weights(model, model_type: str, source: str, verbose=True):
    """
    Load encoder weights (gat_encoder/hg_encoder + lstm_encoder) from
    pretrained BART or Shanghai checkpoint into the BKK model.

    The prediction head (od_decoder) is NOT loaded — it is city-specific
    and will be trained from scratch on BKK data.

    If source='both', BART weights are loaded first, then Shanghai weights
    are averaged in for the shared encoder layers.
    """
    encoder_prefixes = ("gat_encoder", "hg_encoder", "lstm_encoder")

    def _ckpt_path(dataset):
        if model_type == "gat":
            return BART_GAT_CKPT if dataset == "bart" else SHANGHAI_GAT_CKPT
        else:
            return BART_HG_CKPT if dataset == "bart" else SHANGHAI_HG_CKPT

    def _load_one(dataset):
        path = _ckpt_path(dataset)
        if not os.path.exists(path):
            if verbose:
                print(f"  [warning] Checkpoint not found: {path}")
            return None
        ckpt = torch.load(path, map_location="cpu")
        # Handle both checkpoint formats:
        # - train_bart.py format:     {'encoder_state': ..., 'full_state': ...}
        # - train_shanghai.py format: same
        if "encoder_state" in ckpt:
            return ckpt["encoder_state"]
        elif "full_state" in ckpt:
            # Extract encoder layers from full state
            return {k: v for k, v in ckpt["full_state"].items()
                    if any(k.startswith(p) for p in encoder_prefixes)}
        else:
            if verbose:
                print(f"  [warning] Unknown checkpoint format in {path}")
            return None

    if source == "both":
        state_bart     = _load_one("bart")
        state_shanghai = _load_one("shanghai")
        if state_bart is None and state_shanghai is None:
            if verbose:
                print("  [warning] No pretrained weights found — training from scratch")
            return model
        elif state_bart is None:
            combined = state_shanghai
        elif state_shanghai is None:
            combined = state_bart
        else:
            # Average encoder weights from both datasets
            combined = {}
            for key in state_bart:
                if key in state_shanghai:
                    # Only average if shapes match (they should for encoder layers)
                    if state_bart[key].shape == state_shanghai[key].shape:
                        combined[key] = (state_bart[key] + state_shanghai[key]) / 2.0
                    else:
                        combined[key] = state_bart[key]  # prefer BART on shape mismatch
                else:
                    combined[key] = state_bart[key]
            if verbose:
                print(f"  Averaged encoder weights from BART + Shanghai")
    else:
        combined = _load_one(source)
        if combined is None:
            if verbose:
                print("  [warning] No pretrained weights found — training from scratch")
            return model

    # Load into model — strict=False allows missing prediction head keys
    current_state = model.state_dict()
    loaded, skipped = 0, 0
    for key, value in combined.items():
        if key in current_state and current_state[key].shape == value.shape:
            current_state[key] = value
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(current_state, strict=False)
    if verbose:
        print(f"  Loaded {loaded} encoder parameter tensors "
              f"({skipped} skipped — shape mismatch or missing)")
    return model


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_encoder(model):
    """Freeze all encoder layers — only prediction head trains."""
    encoder_prefixes = ("gat_encoder", "hg_encoder", "lstm_encoder")
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in encoder_prefixes):
            param.requires_grad = False
            frozen += 1
    return frozen


def unfreeze_lstm(model):
    """Unfreeze LSTM encoder only."""
    unfrozen = 0
    for name, param in model.named_parameters():
        if name.startswith("lstm_encoder"):
            param.requires_grad = True
            unfrozen += 1
    return unfrozen


def unfreeze_all(model):
    """Unfreeze everything."""
    for param in model.parameters():
        param.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# BKK scenario loading (mirrors train.py)
# ---------------------------------------------------------------------------

def load_bkk_scenarios(zone_ids, device, in_channels,
                         gtfs_features=None, real_weight=3.0):
    scenarios = []
    print("\nLoading BKK scenarios...")

    m2_base = None
    if os.path.exists(M2_BASE_KK) and os.path.exists(M2_DEV_KK):
        m2_base = load_od_matrix_with_header(M2_BASE_KK)
        m2_dev  = load_od_matrix_with_header(M2_DEV_KK)
        m2_diff = m2_dev - m2_base
        scenarios.append({
            "name":   "M2 extension",
            "weight": real_weight,
            "x_seq":  [
                od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                od_matrix_to_zone_features(m2_dev,  in_channels, gtfs_features).to(device),
            ],
            "scenario_feat": build_scenario_features(
                "metro_extension", get_affected_zones(m2_diff, zone_ids)
            ).to(device),
            "target": diff_to_target(m2_diff, zone_ids, device),
        })
        print("  M2 extension")
    else:
        print(f"  [missing] M2 data: {M2_BASE_KK}")
        sys.exit(1)

    if os.path.exists(BUS35_KK) and os.path.exists(BUS35_DIFF_KK):
        try:
            bus_kk   = load_od_matrix_with_header(BUS35_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            bus_diff = load_od_matrix_with_header(BUS35_DIFF_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            scenarios.append({
                "name":   "Bus 35",
                "weight": real_weight,
                "x_seq":  [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    od_matrix_to_zone_features(bus_kk,  in_channels, gtfs_features).to(device),
                ],
                "scenario_feat": build_scenario_features(
                    "bus_new", get_affected_zones(bus_diff, zone_ids)
                ).to(device),
                "target": diff_to_target(bus_diff, zone_ids, device),
            })
            print("  Bus 35 Pesterzsébet")
        except Exception as e:
            print(f"  [error] Bus 35: {e}")

    # M1 = validation (held out)
    val_scenario = None
    if os.path.exists(M1_KK) and os.path.exists(M1_DIFF_KK):
        try:
            m1_kk   = load_od_matrix_with_header(M1_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            m1_diff = load_od_matrix_with_header(M1_DIFF_KK).reindex(
                index=zone_ids, columns=zone_ids).fillna(0)
            val_scenario = {
                "name":   "M1 extension",
                "weight": 1.0,
                "x_seq":  [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    od_matrix_to_zone_features(m1_kk,   in_channels, gtfs_features).to(device),
                ],
                "scenario_feat": build_scenario_features(
                    "metro_extension", get_affected_zones(m1_diff, zone_ids)
                ).to(device),
                "target": diff_to_target(m1_diff, zone_ids, device),
            }
            print("  M1 extension (val)")
        except Exception as e:
            print(f"  [error] M1: {e}")

    # Synthetic scenarios
    metadata = os.path.join(SYNTHETIC_DIR, "metadata.json")
    if os.path.exists(metadata):
        syn = load_scenarios(SYNTHETIC_DIR, zone_ids)
        for diff, meta in syn:
            affected = (meta.get("affected_zones") or
                        meta.get("corridor_zones") or
                        meta.get("stops") or [])
            x_dev = od_matrix_to_zone_features(
                m2_base + diff, in_channels, gtfs_features
            )
            scenarios.append({
                "name":   meta["scenario_id"],
                "weight": 1.0,
                "x_seq":  [
                    od_matrix_to_zone_features(m2_base, in_channels, gtfs_features).to(device),
                    x_dev.to(device),
                ],
                "scenario_feat": build_scenario_features(
                    meta["type"], affected
                ).to(device),
                "target": diff_to_target(diff, zone_ids, device),
            })
        print(f"  {len(syn)} synthetic scenarios")

    # Normalise targets
    for s in scenarios + ([val_scenario] if val_scenario else []):
        t     = s["target"]
        t_std = t.std().clamp(min=1e-6)
        s["target_std"] = t_std
        s["target"]     = t / t_std

    return scenarios, val_scenario


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_transfer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model.upper()}+LSTM | Source: {args.source}")

    # BKK data setup
    m2_base  = load_od_matrix_with_header(M2_BASE_KK)
    zone_ids = m2_base.index.tolist()
    print(f"BKK zones: {len(zone_ids)}")

    gtfs_features = None
    if args.use_gtfs and os.path.exists(GTFS_ZIP):
        try:
            gtfs_features = build_gtfs_zone_features(
                GTFS_ZIP, zone_ids,
                zones_shp_path=ZONES_SHP if os.path.exists(ZONES_SHP) else None,
            )
        except Exception as e:
            print(f"GTFS build failed ({e}) — using 16-dim features")

    in_channels = NUM_FEATURES if gtfs_features else 16

    # Build BKK model
    if args.model == "gat":
        from models.gat_lstm import GATLSTMModel, Config, build_zone_graph
        cfg = Config()
        cfg.GAT_IN_CHANNELS = in_channels
        model = GATLSTMModel(cfg).to(device)
        save_path = TRANSFER_GAT_CKPT

        try:
            import geopandas as gpd
            gdf = gpd.read_file(ZONES_SHP).to_crs(epsg=4326)
            gdf["NO"] = gdf["NO"].astype(int)
        except Exception:
            gdf = None
        edge_index = build_zone_graph(zone_ids, gdf=gdf).to(device)

    else:
        from models.hypergraph_lstm import (
            HypergraphLSTMModel, HypergraphConfig, build_incidence_matrix
        )
        cfg = HypergraphConfig()
        cfg.HG_IN_CHANNELS = in_channels
        model = HypergraphLSTMModel(cfg).to(device)
        save_path = TRANSFER_HG_CKPT
        edge_index = None

        gtfs_routes = None
        if os.path.exists(GTFS_ZIP):
            try:
                import zipfile, geopandas as gpd
                from scipy.spatial import cKDTree
                gdf = gpd.read_file(ZONES_SHP)
                gdf["NO"] = gdf["NO"].astype(int)
                gdf_proj  = gdf.to_crs(epsg=23700)
                centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)
                gdf["lon"] = centroids.x
                gdf["lat"] = centroids.y
                with zipfile.ZipFile(GTFS_ZIP) as z:
                    stops      = pd.read_csv(z.open("stops.txt"))
                    stop_times = pd.read_csv(z.open("stop_times.txt"))
                    trips      = pd.read_csv(z.open("trips.txt"))
                gi     = gdf.set_index("NO")
                coords = np.column_stack([
                    gi.reindex(zone_ids)["lon"].fillna(19.0),
                    gi.reindex(zone_ids)["lat"].fillna(47.5),
                ])
                _, idx = cKDTree(coords).query(
                    stops[["stop_lon", "stop_lat"]].values, k=1
                )
                stops["zone_id"] = [zone_ids[i] for i in idx]
                st = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id")
                st["zone_id"] = st["stop_id"].map(
                    dict(zip(stops["stop_id"], stops["zone_id"]))
                )
                gtfs_routes = (
                    st.groupby("route_id")["zone_id"]
                    .apply(lambda x: list(set(x.dropna())))
                    .to_dict()
                )
            except Exception as e:
                print(f"GTFS error: {e}")
        H = build_incidence_matrix(zone_ids, gtfs_routes=gtfs_routes)
        model.set_hypergraph(H)

    # Load pretrained encoder weights
    print(f"\nLoading pretrained encoder from: {args.source}")
    model = load_encoder_weights(model, args.model, args.source, verbose=True)

    # Load BKK scenarios
    train_scenarios, val_scenario = load_bkk_scenarios(
        zone_ids, device, in_channels,
        gtfs_features=gtfs_features,
        real_weight=args.real_weight,
    )

    if args.model == "gat":
        for s in train_scenarios + ([val_scenario] if val_scenario else []):
            s["edge_index"] = edge_index

    print(f"\nTrain scenarios: {len(train_scenarios)}")
    print(f"Val scenario: {val_scenario['name'] if val_scenario else 'None'}")

    # ── Phase 1: Frozen encoder ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Phase 1: Frozen encoder — training head only")
    frozen = freeze_encoder(model)
    print(f"  Frozen: {frozen} tensors | Trainable params: {count_trainable(model):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs, eta_min=args.lr_head * 0.1
    )

    best_val_loss = float("inf")
    save_path_phase1 = save_path.replace(".pt", "_phase1.pt")

    model, best_val_loss = _train_phase(
        model, args.model, optimizer, scheduler,
        train_scenarios, val_scenario,
        args.phase1_epochs, args.patience,
        save_path_phase1, best_val_loss,
        phase_name="Phase 1",
    )

    # ── Phase 2: Unfreeze LSTM ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Phase 2: Unfreezing LSTM encoder")
    unfrozen = unfreeze_lstm(model)
    print(f"  Unfrozen: {unfrozen} tensors | Trainable params: {count_trainable(model):,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_lstm, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase2_epochs, eta_min=args.lr_lstm * 0.1
    )

    save_path_phase2 = save_path.replace(".pt", "_phase2.pt")
    model, best_val_loss = _train_phase(
        model, args.model, optimizer, scheduler,
        train_scenarios, val_scenario,
        args.phase2_epochs, args.patience,
        save_path_phase2, best_val_loss,
        phase_name="Phase 2",
    )

    # ── Phase 3: Full fine-tuning ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Phase 3: Full fine-tuning (low LR)")
    unfreeze_all(model)
    print(f"  Trainable params: {count_trainable(model):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_full, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase3_epochs, eta_min=args.lr_full * 0.1
    )

    model, best_val_loss = _train_phase(
        model, args.model, optimizer, scheduler,
        train_scenarios, val_scenario,
        args.phase3_epochs, args.patience,
        save_path, best_val_loss,
        phase_name="Phase 3 (final)",
    )

    print(f"\n{'='*50}")
    print(f"Transfer learning complete")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {save_path}")

    # Compare with baseline (original train.py checkpoint)
    baseline_path = GAT_CHECKPOINT if args.model == "gat" else HG_CHECKPOINT
    if os.path.exists(baseline_path) and val_scenario is not None:
        print(f"\n── Comparison with baseline (train.py) ──")
        try:
            baseline_ckpt = torch.load(baseline_path, map_location=device)
            baseline_val  = baseline_ckpt.get("best_val_loss", "N/A")
            print(f"  Baseline val loss:  {baseline_val}")
            print(f"  Transfer val loss:  {best_val_loss:.4f}")
            if isinstance(baseline_val, float):
                delta = best_val_loss - baseline_val
                sign  = "+" if delta > 0 else ""
                print(f"  Delta:              {sign}{delta:.4f} "
                      f"({'worse' if delta > 0 else 'better'})")
        except Exception as e:
            print(f"  Could not load baseline: {e}")


def _train_phase(model, model_type, optimizer, scheduler,
                  train_scenarios, val_scenario,
                  n_epochs, patience, save_path, best_val_loss,
                  phase_name=""):
    """Single training phase — shared by all 3 phases."""
    patience_counter = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_losses = []

        for s in np.random.permutation(len(train_scenarios)):
            sc = train_scenarios[int(s)]
            optimizer.zero_grad()

            if model_type == "gat":
                pred = model(sc["x_seq"], sc["edge_index"], sc["scenario_feat"])
            else:
                pred = model(sc["x_seq"], sc["scenario_feat"])

            loss = combined_loss(pred, sc["target"]) * sc["weight"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item() / sc["weight"])

        train_loss = float(np.mean(epoch_losses))
        scheduler.step()

        if val_scenario is not None:
            model.eval()
            with torch.no_grad():
                if model_type == "gat":
                    val_pred = model(val_scenario["x_seq"],
                                     val_scenario["edge_index"],
                                     val_scenario["scenario_feat"])
                else:
                    val_pred = model(val_scenario["x_seq"],
                                     val_scenario["scenario_feat"])
                val_loss = combined_loss(val_pred, val_scenario["target"]).item()
                val_mae  = F.l1_loss(val_pred, val_scenario["target"]).item()

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_val_loss":    best_val_loss,
                    "phase":            phase_name,
                    "epoch":            epoch,
                }, save_path)

            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch <= 3:
                print(f"  [{phase_name}] Ep {epoch:3d}/{n_epochs} | "
                      f"train={train_loss:.4f} | val={val_loss:.4f} | "
                      f"mae={val_mae:.4f} | "
                      f"{'✓' if improved else ' '}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
        else:
            if epoch % 10 == 0:
                print(f"  [{phase_name}] Ep {epoch:3d}/{n_epochs} | "
                      f"train={train_loss:.4f}")

    return model, best_val_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning: BART/Shanghai → BKK")
    parser.add_argument("--model",         choices=["gat", "hypergraph"], default="gat")
    parser.add_argument("--source",        choices=["bart", "shanghai", "both"],
                        default="bart",
                        help="Which pretrained checkpoint(s) to load encoder from")
    parser.add_argument("--phase1_epochs", type=int,   default=50,
                        help="Epochs with frozen encoder (head only)")
    parser.add_argument("--phase2_epochs", type=int,   default=50,
                        help="Epochs with unfrozen LSTM")
    parser.add_argument("--phase3_epochs", type=int,   default=100,
                        help="Epochs full fine-tuning")
    parser.add_argument("--lr_head",       type=float, default=1e-3,
                        help="LR for phase 1 (head only)")
    parser.add_argument("--lr_lstm",       type=float, default=3e-4,
                        help="LR for phase 2 (LSTM unfrozen)")
    parser.add_argument("--lr_full",       type=float, default=1e-4,
                        help="LR for phase 3 (full fine-tuning)")
    parser.add_argument("--patience",      type=int,   default=15)
    parser.add_argument("--real_weight",   type=float, default=3.0)
    parser.add_argument("--use_gtfs",      action="store_true", default=True)
    parser.add_argument("--no_gtfs",       dest="use_gtfs", action="store_false")
    args = parser.parse_args()

    print(f"Transfer: {args.source} → BKK | Model: {args.model.upper()}")
    print(f"Phases: {args.phase1_epochs}+{args.phase2_epochs}+{args.phase3_epochs} epochs")
    print(f"LRs: head={args.lr_head} | lstm={args.lr_lstm} | full={args.lr_full}")
    run_transfer(args)