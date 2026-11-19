"""Tests for the minimal ZIP-EXR decoder's byte transform (predictor + interleave)."""

import numpy as np

from sensorfault.data.exr_zip import _unpredict_deinterleave


def _forward(raw: bytes) -> bytes:
    # OpenEXR ZIP pre-processing: interleave even/odd bytes, then delta-encode with +128
    a = np.frombuffer(raw, dtype=np.uint8)
    t = np.concatenate([a[0::2], a[1::2]]).astype(np.int64)
    d = np.empty_like(t)
    d[0] = t[0]
    d[1:] = (t[1:] - t[:-1] + 128) % 256
    return d.astype(np.uint8).tobytes()


def test_roundtrip_of_predictor_and_interleave():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 64, 1001):
        raw = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        assert _unpredict_deinterleave(_forward(raw)) == raw
    floats = np.linspace(200.0, 400.0, 96, dtype=np.float32).tobytes()
    assert _unpredict_deinterleave(_forward(floats)) == floats
