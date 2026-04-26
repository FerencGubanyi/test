"""
utils/synthetic_scenarios.py
Synthetic ΔOD matrix generator for BKK scenario augmentation.

Key improvement over v1: all generators now use _ipf_balance() with the
baseline OD matrix instead of the simpler _enforce_conservation(). This
enforces proper row AND column marginal conservation (same trip totals per
zone as baseline), producing synthetic deltas that are structurally closer
to real VISUM outputs.
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Dict, Tuple, Optional


def extract_scenario_profile(diff_matrix: pd.DataFrame,
                              scenario_type: str = 'reference') -> Dict:
    vals     = diff_matrix.values.flatten()
    nonzero  = vals[np.abs(vals) > 0.01]
    zone_net = diff_matrix.sum(axis=1) + diff_matrix.sum(axis=0)
    profile  = {
        'scenario_type':  scenario_type,
        'n_zones':        diff_matrix.shape[0],
        'sparsity':       1 - len(nonzero) / len(vals),
        'val_mean':       float(nonzero.mean()) if len(nonzero) > 0 else 0,
        'val_std':        float(nonzero.std())  if len(nonzero) > 0 else 1,
        'val_p25':        float(np.percentile(nonzero, 25)) if len(nonzero) > 0 else 0,
        'val_p75':        float(np.percentile(nonzero, 75)) if len(nonzero) > 0 else 0,
        'pos_ratio':      float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.5,
        'zone_net_std':   float(zone_net.std()),
        'zone_net_max':   float(zone_net.abs().max()),
        'affected_zones': int((zone_net.abs() > zone_net.abs().quantile(0.8)).sum()),
        'total_net':      float(vals.sum()),
    }
    print(f'Profile ({scenario_type}):')
    print(f'  Sparsity:       {profile["sparsity"]:.1%}')
    print(f'  Affected zones: {profile["affected_zones"]}')
    print(f'  Value std:      {profile["val_std"]:.3f}')
    return profile


class SyntheticScenarioGenerator:

    SIZE_PARAMS = {
        'bus_new': {
            'small':  {'radius': 2.0, 'n_zones': 15, 'magnitude':  0.3},
            'medium': {'radius': 4.0, 'n_zones': 35, 'magnitude':  0.6},
            'large':  {'radius': 6.0, 'n_zones': 60, 'magnitude':  1.0},
        },
        'tram_extension': {
            'small':  {'n_stops': 2, 'corridor_km': 1.5, 'magnitude':  0.4},
            'medium': {'n_stops': 4, 'corridor_km': 2.5, 'magnitude':  0.7},
            'large':  {'n_stops': 6, 'corridor_km': 3.5, 'magnitude':  1.1},
        },
        'stop_closure': {
            'small':  {'radius': 1.5, 'n_zones': 10, 'magnitude': -0.2},
            'medium': {'radius': 3.0, 'n_zones': 25, 'magnitude': -0.5},
            'large':  {'radius': 5.0, 'n_zones': 45, 'magnitude': -0.8},
        },
        'metro_extension': {
            'small':  {'n_new': 1, 'magnitude': 0.4},
            'medium': {'n_new': 2, 'magnitude': 0.7},
            'large':  {'n_new': 3, 'magnitude': 1.1},
        },
        'bus_frequency_increase': {
            'small':  {'freq_mult': 1.5, 'elasticity': 0.35},
            'medium': {'freq_mult': 2.0, 'elasticity': 0.50},
            'large':  {'freq_mult': 3.0, 'elasticity': 0.65},
        },
        'parallel_route_addition': {
            'small':  {'abstraction': 0.08, 'magnitude': 0.3},
            'medium': {'abstraction': 0.18, 'magnitude': 0.6},
            'large':  {'abstraction': 0.28, 'magnitude': 1.0},
        },
    }

    def __init__(self, zone_ids: List[int], profile: Dict,
                 baseline: Optional[np.ndarray] = None,
                 gdf=None, seed: int = 42):
        self.zone_ids    = zone_ids
        self.n           = len(zone_ids)
        self.profile     = profile
        self.rng         = np.random.default_rng(seed)
        self.zone_to_idx = {z: i for i, z in enumerate(zone_ids)}
        # Baseline OD matrix — passed to _ipf_balance for proper conservation
        self.baseline    = baseline

        if gdf is not None:
            gi = gdf.set_index('NO')
            self.centroids = np.column_stack([
                gi.reindex(zone_ids)['centroid_lon'].fillna(19.0).values,
                gi.reindex(zone_ids)['centroid_lat'].fillna(47.5).values,
            ])
        else:
            rng2 = np.random.default_rng(seed + 1)
            self.centroids = np.column_stack([
                rng2.uniform(18.8, 19.3, self.n),
                rng2.uniform(47.3, 47.7, self.n),
            ])

    def _nearby(self, center_idx: int, radius_km: float, n: int) -> np.ndarray:
        c     = self.centroids[center_idx]
        dlat  = (self.centroids[:, 1] - c[1]) * 111.0
        dlon  = (self.centroids[:, 0] - c[0]) * 111.0 * np.cos(np.radians(c[1]))
        dists = np.sqrt(dlat**2 + dlon**2)
        within = np.where(dists < radius_km)[0]
        if len(within) < n:
            within = np.argsort(dists)[:n]
        return self.rng.choice(within, size=min(n, len(within)), replace=False)

    def _enforce_conservation(self, delta: np.ndarray) -> np.ndarray:
        """Fallback: distribute residual among non-zero cells."""
        total = delta.sum()
        mask  = np.abs(delta) > 1e-6
        if abs(total) > 0.01 and mask.sum() > 0:
            delta[mask] -= total / mask.sum()
        return delta

    def _ipf_balance(self, delta: np.ndarray,
                     baseline: Optional[np.ndarray] = None,
                     max_iter: int = 40, tol: float = 1e-5) -> np.ndarray:
        """
        Iterative Proportional Fitting — enforces both row and column
        conservation so the modified OD has the same marginals as baseline.
        Falls back to _enforce_conservation if no baseline is available.
        """
        bl = baseline if baseline is not None else self.baseline
        if bl is None:
            return self._enforce_conservation(delta)

        modified = np.maximum(bl + delta, 0.0)
        np.fill_diagonal(modified, 0.0)
        O = bl.sum(axis=1)
        D = bl.sum(axis=0)

        for _ in range(max_iter):
            rs = modified.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                modified *= np.where(rs > 1e-9, O / rs, 1.0)[:, None]
            np.fill_diagonal(modified, 0.0)
            cs = modified.sum(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                modified *= np.where(cs > 1e-9, D / cs, 1.0)[None, :]
            np.fill_diagonal(modified, 0.0)
            if (np.abs(modified.sum(axis=1) - O).max() < tol and
                    np.abs(modified.sum(axis=0) - D).max() < tol):
                break

        return modified - bl

    def _finalize(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply IPF balancing with the stored baseline OD matrix.
        If no baseline was provided at init, falls back to simple conservation.
        Called at the end of every scenario generator.
        """
        return self._ipf_balance(delta)

    def generate_bus_new(self, scenario_id: int,
                          size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p          = self.SIZE_PARAMS['bus_new'][size]
        center_idx = self.rng.integers(0, self.n)
        affected   = self._nearby(center_idx, p['radius'], p['n_zones'])
        scale      = self.profile['val_std'] * p['magnitude']

        delta = np.zeros((self.n, self.n))
        for i in affected:
            for j in affected:
                if i != j:
                    dist     = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    strength = np.exp(-dist * 10) * scale
                    delta[i, j] += self.rng.normal(strength, max(strength * 0.3, 1e-6))

        delta = self._finalize(delta)
        meta  = {
            'scenario_id':    f'syn_bus_new_{size}_{scenario_id:02d}',
            'type':           'bus_new',
            'size':           size,
            'center_zone':    int(self.zone_ids[center_idx]),
            'affected_zones': [int(self.zone_ids[i]) for i in affected],
            'n_affected':     len(affected),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_tram_extension(self, scenario_id: int,
                                 size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p         = self.SIZE_PARAMS['tram_extension'][size]
        start_idx = self.rng.integers(0, self.n)
        angle     = self.rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])

        stop_indices = [start_idx]
        for k in range(1, p['n_stops']):
            target = self.centroids[start_idx] + direction * k * 0.02
            dists  = np.linalg.norm(self.centroids - target, axis=1)
            stop_indices.append(int(np.argmin(dists)))

        corridor = set()
        for si in stop_indices:
            corridor.update(self._nearby(si, p['corridor_km'], 20).tolist())
        corridor = list(corridor)

        scale = self.profile['val_std'] * p['magnitude']
        delta = np.zeros((self.n, self.n))
        for i in corridor:
            for j in corridor:
                if i != j:
                    end_dist = np.linalg.norm(
                        self.centroids[i] - self.centroids[stop_indices[-1]]
                    )
                    strength = scale * np.exp(-end_dist * 5)
                    delta[i, j] += self.rng.normal(strength, max(strength * 0.25, 1e-6))

        delta = self._finalize(delta)
        meta  = {
            'scenario_id':    f'syn_tram_ext_{size}_{scenario_id:02d}',
            'type':           'tram_extension',
            'size':           size,
            'stops':          [int(self.zone_ids[i]) for i in stop_indices],
            'corridor_zones': [int(self.zone_ids[i]) for i in corridor],
            'n_affected':     len(corridor),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_stop_closure(self, scenario_id: int,
                               size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p          = self.SIZE_PARAMS['stop_closure'][size]
        center_idx = self.rng.integers(0, self.n)
        affected   = self._nearby(center_idx, p['radius'], p['n_zones'])
        scale      = abs(self.profile['val_std'] * p['magnitude'])

        delta = np.zeros((self.n, self.n))
        for i in affected:
            for j in affected:
                if i != j:
                    dist     = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    strength = -np.exp(-dist * 10) * scale
                    delta[i, j] += self.rng.normal(strength, scale * 0.2)

        border = self._nearby(center_idx, p['radius'] * 1.5, p['n_zones'] // 2)
        border = [z for z in border if z not in affected]
        if border and abs(delta.sum()) > 0:
            gain = abs(delta.sum()) / len(border) / self.n
            for i in border:
                delta[i, :] += gain

        delta = self._finalize(delta)
        meta  = {
            'scenario_id':    f'syn_stop_closure_{size}_{scenario_id:02d}',
            'type':           'stop_closure',
            'size':           size,
            'center_zone':    int(self.zone_ids[center_idx]),
            'affected_zones': [int(self.zone_ids[i]) for i in affected],
            'n_affected':     len(affected),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_metro_extension(self, scenario_id: int,
                                  size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p = self.SIZE_PARAMS['metro_extension'][size]

        all_zones    = list(range(self.n))
        corridor_size = self.rng.integers(8, max(9, self.n // 5))
        start        = self.rng.integers(0, self.n)
        corridor     = self._nearby(start, 8.0, int(corridor_size)).tolist()
        if len(corridor) < 3:
            corridor = list(range(min(10, self.n)))

        non_corridor = [z for z in all_zones if z not in set(corridor)]
        n_new        = min(p['n_new'], len(non_corridor))
        if n_new == 0:
            return self._empty(f'syn_metro_ext_{size}_{scenario_id:02d}',
                               'metro_extension', size)

        anchor = corridor[-1]
        dists  = np.abs(np.array(non_corridor) - anchor)
        probs  = 1.0 / (dists + 1.0)
        probs /= probs.sum()
        new_zones = self.rng.choice(non_corridor, size=n_new, replace=False, p=probs)

        magnitude = self.profile['val_std'] * p['magnitude']
        delta     = np.zeros((self.n, self.n))

        for new_z in new_zones:
            attr    = np.abs(np.arange(len(corridor)) - len(corridor)) + 1.0
            weights = attr / attr.sum()
            for src_z, w in zip(corridor, weights):
                gain = magnitude * w + self.rng.exponential(0.5)
                delta[src_z, new_z] += gain
                alts = [z for z in corridor if z != src_z and z != new_z]
                if alts:
                    delta[src_z, self.rng.choice(alts)] -= gain

            bus_alts = [z for z in non_corridor if z != new_z]
            if bus_alts:
                n_bus  = min(3, len(bus_alts))
                chosen = self.rng.choice(bus_alts, size=n_bus, replace=False)
                loss_each = magnitude * 0.4 / n_bus
                for bz in chosen:
                    delta[new_z, bz]               -= loss_each
                    delta[new_z, self.rng.choice(corridor)] += loss_each

        delta = self._finalize(delta)
        meta  = {
            'scenario_id':    f'syn_metro_ext_{size}_{scenario_id:02d}',
            'type':           'metro_extension',
            'size':           size,
            'corridor_zones': [int(self.zone_ids[i]) for i in corridor],
            'affected_zones': [int(self.zone_ids[i]) for i in new_zones],
            'n_affected':     len(corridor) + len(new_zones),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_bus_frequency_increase(
            self, scenario_id: int,
            size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p               = self.SIZE_PARAMS['bus_frequency_increase'][size]
        demand_increase = p['elasticity'] * (p['freq_mult'] - 1.0)

        center_idx = self.rng.integers(0, self.n)
        corridor   = self._nearby(center_idx, 4.0, 20).tolist()
        corr_set   = set(corridor)
        non_corr   = [z for z in range(self.n) if z not in corr_set]

        delta = np.zeros((self.n, self.n))
        for i, z_o in enumerate(corridor):
            for z_d in corridor[i + 1:]:
                for zo, zd in [(z_o, z_d), (z_d, z_o)]:
                    gain = self.profile['val_std'] * demand_increase * \
                           self.rng.uniform(0.8, 1.2)
                    delta[zo, zd] += gain
                    if non_corr:
                        bleed = self.rng.choice(non_corr)
                        delta[zo, bleed] -= gain * 0.25

        delta = self._finalize(delta)
        meta  = {
            'scenario_id':    f'syn_bus_freq_{size}_{scenario_id:02d}',
            'type':           'bus_frequency_increase',
            'size':           size,
            'center_zone':    int(self.zone_ids[center_idx]),
            'corridor_zones': [int(self.zone_ids[i]) for i in corridor],
            'n_affected':     len(corridor),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_parallel_route_addition(
            self, scenario_id: int,
            size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        p = self.SIZE_PARAMS['parallel_route_addition'][size]

        start   = self.rng.integers(0, self.n)
        metro   = self._nearby(start, 8.0, self.rng.integers(8, 16)).tolist()
        if len(metro) < 4:
            metro = list(range(min(12, self.n)))

        n_parallel = max(2, len(metro) // 2)
        seg_start  = self.rng.integers(0, len(metro) - n_parallel)
        parallel   = metro[seg_start: seg_start + n_parallel]

        non_metro = [z for z in range(self.n) if z not in set(metro)]
        n_excl    = self.rng.integers(2, 6)
        exclusive = (self.rng.choice(non_metro,
                                     size=min(n_excl, len(non_metro)),
                                     replace=False).tolist()
                     if non_metro else [])

        magnitude   = self.profile['val_std'] * p['magnitude']
        suppression = p['abstraction'] * 0.25
        delta       = np.zeros((self.n, self.n))

        for i, z_o in enumerate(parallel):
            for z_d in parallel[i + 1:]:
                for zo, zd in [(z_o, z_d), (z_d, z_o)]:
                    suppress = self.profile['val_std'] * suppression
                    delta[zo, zd] -= suppress
                    if exclusive:
                        delta[zo, self.rng.choice(exclusive)] += suppress * 0.6

        for excl_z in exclusive:
            for seg_z in parallel:
                for zo, zd in [(seg_z, excl_z), (excl_z, seg_z)]:
                    gain = magnitude * self.rng.uniform(0.5, 1.5) + 0.5
                    delta[zo, zd] += gain
                    non_par = [z for z in range(self.n)
                               if z not in set(parallel + exclusive) and z != zo]
                    if non_par:
                        delta[zo, self.rng.choice(non_par)] -= gain * 0.7

        delta        = self._finalize(delta)
        all_affected = parallel + exclusive
        meta         = {
            'scenario_id':    f'syn_parallel_{size}_{scenario_id:02d}',
            'type':           'parallel_route_addition',
            'size':           size,
            'corridor_zones': [int(self.zone_ids[i]) for i in parallel],
            'affected_zones': [int(self.zone_ids[i]) for i in exclusive],
            'n_affected':     len(all_affected),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def _empty(self, scenario_id, stype, size) -> Tuple[pd.DataFrame, Dict]:
        delta = np.zeros((self.n, self.n))
        meta  = {'scenario_id': scenario_id, 'type': stype, 'size': size,
                 'affected_zones': [], 'n_affected': 0}
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_batch(self, n_per_type: int = 30) -> List[Tuple[pd.DataFrame, Dict]]:
        sizes      = ['small', 'medium', 'large']
        generators = {
            'bus_new':                 self.generate_bus_new,
            'tram_extension':          self.generate_tram_extension,
            'stop_closure':            self.generate_stop_closure,
            'metro_extension':         self.generate_metro_extension,
            'bus_frequency_increase':  self.generate_bus_frequency_increase,
            'parallel_route_addition': self.generate_parallel_route_addition,
        }
        results = []
        counter = 0
        print(f'Generating {n_per_type} x {len(generators)} = '
              f'{n_per_type * len(generators)} synthetic scenarios...')
        for type_name, fn in generators.items():
            for i in range(n_per_type):
                diff, meta = fn(counter, sizes[i % len(sizes)])
                results.append((diff, meta))
                counter += 1
            print(f'  {type_name}: {n_per_type} done')
        return results


def validate_synthetic(real_diff: pd.DataFrame,
                        synthetic_diffs: List[pd.DataFrame]) -> pd.DataFrame:
    def stats(df, name):
        v  = df.values.flatten()
        nz = v[np.abs(v) > 0.01]
        zn = df.sum(axis=1) + df.sum(axis=0)
        return {
            'scenario':     name,
            'sparsity':     round(1 - len(nz) / len(v), 4),
            'val_std':      round(nz.std()  if len(nz) > 0 else 0, 4),
            'val_mean':     round(nz.mean() if len(nz) > 0 else 0, 4),
            'total_net':    round(v.sum(), 2),
            'zone_net_std': round(zn.std(), 4),
            'n_affected':   int((zn.abs() > zn.abs().quantile(0.8)).sum()),
        }
    rows = [stats(real_diff, 'real M2')]
    for i, syn in enumerate(synthetic_diffs):
        rows.append(stats(syn, f'synthetic {i+1}'))
    return pd.DataFrame(rows).set_index('scenario')


def save_scenarios(scenarios: List[Tuple[pd.DataFrame, Dict]],
                   output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for diff, meta in scenarios:
        diff.to_csv(os.path.join(output_dir, f'{meta["scenario_id"]}_diff.csv'))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump([m for _, m in scenarios], f, indent=2, ensure_ascii=False)
    print(f'Saved {len(scenarios)} scenarios to {output_dir}')


def load_scenarios(output_dir: str,
                   zone_ids: List[int]) -> List[Tuple[pd.DataFrame, Dict]]:
    with open(os.path.join(output_dir, 'metadata.json')) as f:
        metas = json.load(f)
    results = []
    for meta in metas:
        diff = pd.read_csv(
            os.path.join(output_dir, f'{meta["scenario_id"]}_diff.csv'),
            index_col=0
        )
        diff.index   = diff.index.astype(int)
        diff.columns = diff.columns.astype(int)
        results.append((diff, meta))
    print(f'Loaded {len(results)} scenarios')
    return results