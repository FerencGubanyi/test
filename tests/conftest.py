# tests/conftest.py
"""
Shared pytest configuration and fixtures.

Fixtures defined here are available to all test modules automatically.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Make project root importable from any test
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


#                              
# Markers
#                              

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (skipped with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require a CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests that require real data files"
    )


#                              
# Common fixtures
#                              

@pytest.fixture(scope="session")
def rng_session():
    """Session-scoped RNG — shared across all tests in the session."""
    return np.random.default_rng(42)


@pytest.fixture
def tiny_od():
    """5-zone OD matrix — fastest possible test input."""
    rng = np.random.default_rng(0)
    od = rng.exponential(50, (5, 5))
    np.fill_diagonal(od, 0)
    return od


@pytest.fixture
def small_od():
    """40-zone OD matrix — standard test input."""
    rng = np.random.default_rng(42)
    od = rng.exponential(80, (40, 40))
    np.fill_diagonal(od, 0)
    return od


@pytest.fixture
def medium_od():
    """100-zone OD matrix — closer to real Budapest TAZ count."""
    rng = np.random.default_rng(99)
    od = rng.exponential(100, (100, 100))
    np.fill_diagonal(od, 0)
    return od


@pytest.fixture
def small_hyperedges(small_od):
    """20 random hyperedges over the small OD zone set."""
    rng = np.random.default_rng(7)
    N = small_od.shape[0]
    return [
        list(rng.choice(N, size=int(rng.integers(3, 12)), replace=False))
        for _ in range(20)
    ]


@pytest.fixture
def small_zone_features(small_od):
    """22-dim zone feature matrix for the small OD."""
    rng = np.random.default_rng(3)
    N = small_od.shape[0]
    feats = np.abs(rng.standard_normal((N, 22)))
    feats[:, 0] = small_od.sum(axis=1)   # total outflow
    feats[:, 1] = small_od.sum(axis=0)   # total inflow
    return feats
