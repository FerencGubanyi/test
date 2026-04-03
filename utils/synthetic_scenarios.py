"""
Synthetic OD Scenario Generator

Due to the limited number of real VISUM scenarios, we generate synthetic ΔOD
matrices based on the statistical profile of the existing M2 scenario.

Generált típusok:
  - bus_new:        new busline creation
  - tram_extension: extensoin of already defined tram line
  - stop_closure:   stop closure/openings
"""

import numpy as np
import pandas as pd
import os
import json
from typing import List, Dict, Tuple, Optional


def extract_scenario_profile(diff_matrix: pd.DataFrame,
                              scenario_type: str = 'reference') -> Dict:
    """
    Load data from generated scenarios
    """
    vals     = diff_matrix.values.flatten()
    nonzero  = vals[np.abs(vals) > 0.01]
    zone_net = diff_matrix.sum(axis=1) + diff_matrix.sum(axis=0)

    profile = {
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

    print(f'Referance profil ({scenario_type}):')
    print(f'  Sparsity:       {profile["sparsity"]:.1%}')
    print(f'  Touched zones: {profile["affected_zones"]}')
    print(f'  Values std:      {profile["val_std"]:.3f}')
    print(f'  Net changes: {profile["total_net"]:.1f}')
    return profile


class SyntheticScenarioGenerator:
    """Synthetic ΔOD matrix generator.
    It preserves the statistical characteristics of the real M2 scenario,
    but with different spatial location and strength.
    """

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
    }

    def __init__(self, zone_ids: List[int], profile: Dict,
                 gdf=None, seed: int = 42):
        self.zone_ids    = zone_ids
        self.n           = len(zone_ids)
        self.profile     = profile
        self.rng         = np.random.default_rng(seed)
        self.zone_to_idx = {z: i for i, z in enumerate(zone_ids)}

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
        """
        Forgalommegmaradás — csak a már nem-nulla cellákban korrigál,
        nem szórja szét az egész mátrixra.
        """
        total = delta.sum()
        nonzero_mask = np.abs(delta) > 1e-6
        n_nonzero = nonzero_mask.sum()
        
        if abs(total) > 0.01 and n_nonzero > 0:
            delta[nonzero_mask] -= total / n_nonzero
        
        return delta

    def generate_bus_new(self, scenario_id: int,
                          size: str = 'medium') -> Tuple[pd.DataFrame, Dict]:
        """
        Simulation of the impact of a new bus line.
        Traffic increases between the affected zones, with a small shift from parallel ines.
        """
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

        delta = self._enforce_conservation(delta)
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
        """
        Simulation of tram line extension.
        """
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
                    end_dist = np.linalg.norm(self.centroids[i] - self.centroids[stop_indices[-1]])
                    strength = scale * np.exp(-end_dist * 5)
                    delta[i, j] += self.rng.normal(strength, max(strength * 0.25, 1e-6))

        delta = self._enforce_conservation(delta)
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
        """
        Simulation of a stop closure.
        Negative local impact, some of the traffic is transferred
        to neighboring stops.
        """
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

        delta = self._enforce_conservation(delta)
        meta  = {
            'scenario_id':    f'syn_stop_closure_{size}_{scenario_id:02d}',
            'type':           'stop_closure',
            'size':           size,
            'center_zone':    int(self.zone_ids[center_idx]),
            'affected_zones': [int(self.zone_ids[i]) for i in affected],
            'n_affected':     len(affected),
        }
        return pd.DataFrame(delta, index=self.zone_ids, columns=self.zone_ids), meta

    def generate_batch(self, n_per_type: int = 3) -> List[Tuple[pd.DataFrame, Dict]]:
        """n_per_type scenario generation (small/medium/large)."""
        sizes   = ['small', 'medium', 'large']
        results = []
        counter = 0
        gens = {
            'bus_new':        self.generate_bus_new,
            'tram_extension': self.generate_tram_extension,
            'stop_closure':   self.generate_stop_closure,
        }
        print(f'Syntetic scenario generation ({n_per_type} × {len(gens)} type)...')
        for type_name, fn in gens.items():
            for i in range(n_per_type):
                diff, meta = fn(counter, sizes[i % len(sizes)])
                results.append((diff, meta))
                counter += 1
                print(f'  ✅ {meta["scenario_id"]} — {meta["n_affected"]} effected zone')
        return results


def validate_synthetic(real_diff: pd.DataFrame,
                        synthetic_diffs: List[pd.DataFrame]) -> pd.DataFrame:
    """Statistical comparison: real vs synthetic scenarios."""
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


def save_scenarios(scenarios: List[Tuple[pd.DataFrame, Dict]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for diff, meta in scenarios:
        diff.to_csv(os.path.join(output_dir, f'{meta["scenario_id"]}_diff.csv'))
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump([m for _, m in scenarios], f, indent=2, ensure_ascii=False)
    print(f'✅ {len(scenarios)} scenario saved: {output_dir}')


def load_scenarios(output_dir: str, zone_ids: List[int]) -> List[Tuple[pd.DataFrame, Dict]]:
    with open(os.path.join(output_dir, 'metadata.json')) as f:
        metas = json.load(f)
    results = []
    for meta in metas:
        diff = pd.read_csv(
            os.path.join(output_dir, f'{meta["scenario_id"]}_diff.csv'), index_col=0
        )
        diff.index   = diff.index.astype(int)
        diff.columns = diff.columns.astype(int)
        results.append((diff, meta))
    print(f'✅ {len(results)} scenario loaded')
    return results
