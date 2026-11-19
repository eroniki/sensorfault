"""Shared fixtures. Tests run on CPU, offline, with no datasets."""

from __future__ import annotations

import numpy as np
import pytest

from sensorfault.baselines import base
from sensorfault.types import DepthFrame


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def frame(rng: np.random.Generator) -> DepthFrame:
    depth = rng.uniform(0.5, 2.0, size=(4, 6)).astype(np.float32)
    K = np.array([[5.0, 0, 3.0], [0, 5.0, 2.0], [0, 0, 1.0]])
    return DepthFrame(sensor_id="s0", frame_id=0, depth=depth, K=K, T_wc=np.eye(4))


@pytest.fixture
def clean_registry():
    """Snapshot/restore the global fuser registry so tests do not leak registrations."""
    saved = dict(base._REGISTRY)
    yield base._REGISTRY
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)
