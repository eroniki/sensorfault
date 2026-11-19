"""Power cost of a per-view bias-field nuisance (docs/design-3dgs-bayes.md, section 8.3).

Matched detectability kappa^2 = phi^T M phi with
M = Q^-1 - Q^-1 A (A^T Q^-1 A)^-1 A^T Q^-1 and Q = Sigma_eps + B Sb B^T (Gaussian prior on the field).
A holds the flat-prior nuisance columns [d, 1]; a flat-prior field moves B into A (hard projection).
Matched MDB (alpha = 0.05, beta = 0.1) = 3.24 / kappa, in units of sigma (fault amplitude 1).

Toy: one 32 x 32 view, depth ramp 2-4, a sharp-edged square region fault (f = 0.12) and a smooth
zero-mean warp (rms 1). Field models: smooth DCT basis k x k, or edge-aware piecewise affine in d on
5 segments (4 stripes plus one segment that coincides with the fault or misses it by 4 px). Prior:
flat, or Gaussian with total field rms 0.5 or 0.25 sigma. Runs in about a second on the CPU.
"""

import math

import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

H = W = 32
N = H * W
YY, XX = np.mgrid[0:H, 0:W].astype(float)
DEPTH = (2.0 + 2.0 * YY / H).ravel()
A0 = np.column_stack([DEPTH, np.ones(N)])
Z = 3.24  # z_alpha + z_beta for alpha = 0.05, beta = 0.1
INF = math.inf


def noise_cov(ell: float, nugget: float) -> np.ndarray:
    """(1 - nugget) * Gaussian kernel(ell) + nugget * I, unit variance; ell = 0 gives iid."""
    if ell == 0:
        return np.eye(N)
    p = np.stack([YY.ravel(), XX.ravel()], 1)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    return (1 - nugget) * np.exp(-d2 / (2 * ell**2)) + nugget * np.eye(N)


def dct_basis(k: int) -> np.ndarray:
    """k x k cosine basis without the constant (already in A), each column at rms 1."""
    cols = []
    for i in range(k):
        for j in range(k):
            if i or j:
                b = np.cos(np.pi * i * (YY + 0.5) / H) * np.cos(np.pi * j * (XX + 0.5) / W)
                cols.append((b / np.sqrt((b**2).mean())).ravel())
    return np.stack(cols, 1)


def segment_basis(masks: list[np.ndarray]) -> np.ndarray:
    """Piecewise affine in depth: [indicator, indicator * d] per segment."""
    dd = DEPTH.reshape(H, W)
    return np.stack([c for m in masks for c in (m.ravel() * 1.0, (m * dd).ravel())], 1)


def mdbs(phis, sigma, basis=None, field_rms=None) -> list[float]:
    """Matched MDB for each fault pattern; basis with field_rms=None is a flat prior."""
    a, q = A0, sigma
    if basis is not None and field_rms is None:
        a = np.column_stack([A0, basis])
    elif basis is not None:
        q = sigma + (field_rms**2 / basis.shape[1]) * basis @ basis.T
    c = cho_factor(q)
    g = a.T @ cho_solve(c, a)
    out = []
    for phi in phis:
        qp = cho_solve(c, phi)
        v = a.T @ qp
        k2 = float(phi @ qp - v @ np.linalg.lstsq(g, v, rcond=None)[0])
        out.append(INF if k2 < 1e-9 else Z / math.sqrt(k2))
    return out


SQUARE = np.zeros((H, W), bool)
SQUARE[7:18, 7:18] = True  # region fault, f = 121/1024 = 0.12
SQUARE_MISS = np.zeros((H, W), bool)
SQUARE_MISS[11:22, 11:22] = True  # a segment 4 px off the fault
_WARP = np.cos(np.pi * (XX + 0.5) / W) * np.cos(np.pi * (YY + 0.5) / H)
FAULTS = [SQUARE.ravel() * 1.0, (_WARP / np.sqrt((_WARP**2).mean())).ravel()]
_STRIPES = [np.equal(XX // 8, s) for s in range(4)]  # four 8-px vertical stripes
SEG_ALIGNED = [s & ~SQUARE for s in _STRIPES] + [SQUARE]
SEG_MISSED = [s & ~SQUARE_MISS for s in _STRIPES] + [SQUARE_MISS]

FIELDS = {
    "none": (None, None),
    "smooth3_flat": (dct_basis(3), None),
    "smooth6_flat": (dct_basis(6), None),
    "smooth10_flat": (dct_basis(10), None),
    "smooth6_rms0.5": (dct_basis(6), 0.5),
    "smooth6_rms0.25": (dct_basis(6), 0.25),
    "smooth3_rms0.5": (dct_basis(3), 0.5),
    "edge_aligned_flat": (segment_basis(SEG_ALIGNED), None),
    "edge_aligned_rms0.5": (segment_basis(SEG_ALIGNED), 0.5),
    "edge_missed_flat": (segment_basis(SEG_MISSED), None),
    "edge_missed_rms0.5": (segment_basis(SEG_MISSED), 0.5),
}

# The section 8.3 table: (iid region, iid warp, correlated region, correlated warp), MDB in sigma.
TABLE = {
    "none": (0.317, 0.101, 0.408, 0.486),
    "smooth3_flat": (0.424, INF, 0.413, INF),
    "smooth6_flat": (0.691, INF, 0.418, INF),
    "smooth10_flat": (0.870, INF, 0.431, INF),
    "smooth6_rms0.5": (0.573, 0.292, 0.411, 0.570),
    "smooth6_rms0.25": (0.453, 0.170, 0.409, 0.509),
    "smooth3_rms0.5": (0.406, 0.582, 0.410, 0.751),
    "edge_aligned_flat": (INF, 0.393, INF, 0.581),
    "edge_aligned_rms0.5": (1.612, 0.113, 1.628, 0.490),
    "edge_missed_flat": (0.373, 0.394, 0.412, 0.582),
    "edge_missed_rms0.5": (0.348, 0.112, 0.411, 0.490),
}


@pytest.fixture(scope="module")
def computed() -> dict[str, tuple[float, ...]]:
    iid, corr = noise_cov(0, 0), noise_cov(3.0, 0.2)
    return {
        name: tuple(mdbs(FAULTS, iid, b, r) + mdbs(FAULTS, corr, b, r))
        for name, (b, r) in FIELDS.items()
    }


@pytest.mark.parametrize("name", list(TABLE))
def test_section_8_3_table(computed, name):
    for got, want in zip(computed[name], TABLE[name]):
        if want == INF:
            assert got == INF
        else:
            assert got == pytest.approx(want, abs=1.5e-3)


def test_iid_baselines_match_closed_form(computed):
    f = SQUARE.mean()
    # region: [d, 1] removes the mean and part of the slope, so a little above 3.24/sqrt(n f(1-f))
    assert computed["none"][0] == pytest.approx(Z / math.sqrt(N * f * (1 - f)), rel=0.05)
    # the warp is orthogonal to [d, 1] (zero mean, cosine in x): exactly 3.24/sqrt(n)
    assert computed["none"][1] == pytest.approx(Z / math.sqrt(N), rel=1e-3)


def test_qualitative_claims(computed):
    none, s10, edge = computed["none"], computed["smooth10_flat"], computed["edge_aligned_rms0.5"]
    # 1. correlated noise: even a flat 10x10 smooth field costs < 6% on the sharp-edged region
    assert s10[2] / none[2] < 1.06
    # ... but under iid noise it costs > 2.5x
    assert s10[0] / none[0] > 2.5
    # 2. a smaller unpredictable rms (better cross-fitted mean) recovers warp power
    assert computed["smooth6_rms0.25"][3] < computed["smooth6_rms0.5"][3]
    # 3. edge-aware field aligned with the fault: about 4x MDB even with a calibrated prior
    assert edge[2] / none[2] > 3.5
    assert computed["edge_missed_rms0.5"][2] / none[2] < 1.02
