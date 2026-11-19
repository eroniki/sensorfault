"""Deterministic seed derivation. Never uses Python's salted ``hash()``."""

from __future__ import annotations

import hashlib

import numpy as np


def stable_hash(s: str) -> int:
    """32-bit stable hash of a string (BLAKE2b), identical across processes/platforms."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


def seed_entropy(master: int, sensor_id: str, fault_id: int) -> list:
    return [int(master), stable_hash(sensor_id), int(fault_id)]


def derive_rng(master: int, sensor_id: str, fault_id: int) -> np.random.Generator:
    """np.random.SeedSequence([master, sensor_id_hash, fault_id]) -> Generator."""
    return np.random.default_rng(np.random.SeedSequence(seed_entropy(master, sensor_id, fault_id)))
