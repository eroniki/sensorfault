import numpy as np
import pytest

from sensorfault.stagea import align as alg
from sensorfault.stagea import consensus as cg
from sensorfault.stagea import records as rc
from sensorfault.stagea import repetition as rp
from sensorfault.stagea import synth
from sensorfault.stagea import tiles as tl
from sensorfault.stagea.synth import make_block


@pytest.fixture(scope="module")
def block():
    K, views, _ = make_block(seed=0, n_strips=2, frames_per_strip=6)
    views = tl.assign_strips(views)
    return K, views, rc.build_entities(K, views)


def test_split_roles_never_touch():
    ij = np.stack(np.meshgrid(np.arange(-7, 9), np.arange(-5, 11)), -1).reshape(-1, 2)
    for group in (1, 2, 3):
        roles = tl.tile_role(ij, group=group, buffer=1)
        assert not tl.touching_roles(ij, roles)
        assert (roles == tl.CAL).any() and (roles == tl.HELD).any()
    # a plain checkerboard without buffer would touch
    plain = np.where((ij.sum(1) % 2) == 0, tl.CAL, tl.HELD)
    assert tl.touching_roles(ij, plain)


def test_cell_index_assignment():
    xy = np.array([[0.0, 0.0], [7.99, 0.1], [8.0, -0.01], [-0.01, 63.9]])
    assert tl.cell_index(xy, 8.0).tolist() == [[0, 0], [0, 0], [1, -1], [-1, 7]]


def test_every_frame_has_one_strip(block):
    _, views, _ = block
    assert all("strip_id" in v and "strip_pos" in v for v in views)
    rows = tl.strip_table(views)
    assert sum(r["frames"] for r in rows) == len(views)
    assert all(r["spacing_m"] == pytest.approx(40.0) for r in rows)


def test_records_schema_and_additivity(block, tmp_path):
    K, views, ent = block
    rng = np.random.default_rng(0)
    recs = rc.concat(
        [
            rc.frame_records(K, v, v["depth"] * (1 + 0.01 * rng.normal(size=v["depth"].shape)), ent)
            for v in views
        ]
    )
    assert recs.Lam0.shape == (len(recs.entity), 3, 3) and recs.eps.shape == (len(recs.entity), 3)
    assert np.all(np.linalg.eigvalsh(recs.Lam0) > 0)
    sig = {v["frame"]: 0.01 * (1 + 0.1 * k) for k, v in enumerate(views)}
    joint = rc.fuse(recs, sig)
    half = recs.frame % 2 == 0
    merged = rc.merge_fused(rc.fuse(recs.subset(half), sig), rc.fuse(recs.subset(~half), sig))
    seen = joint[2] > 0
    rel = np.linalg.norm(merged[0][seen] - joint[0][seen], axis=(1, 2)) / np.linalg.norm(
        joint[0][seen], axis=(1, 2)
    )
    assert rel.max() < 1e-6
    np.testing.assert_allclose(merged[1][seen], joint[1][seen], rtol=1e-6, atol=1e-9)
    recs.save(tmp_path / "records.npz")
    z = np.load(tmp_path / "records.npz")
    assert set(z.files) >= {"entity", "frame", "Lam0", "eps", "npix", "shape_rms", "M"}


def test_white_noise_records_are_honest(block):
    K, views, ent = block
    rng = np.random.default_rng(1)
    recs = rc.concat(
        [
            rc.frame_records(K, v, v["depth"] * (1 + 0.01 * rng.normal(size=v["depth"].shape)), ent)
            for v in views
        ]
    )
    q = np.einsum("ri,rij,rj->r", recs.eps, recs.Lam0, recs.eps)
    assert np.sqrt(np.median(q) / 2.366) == pytest.approx(0.01, rel=0.05)


def test_planted_offset_is_recovered(block):
    K, views, ent = block
    v = views[0]
    # move every surface 1 m towards the camera along its normal: dh = +1 => dd = 1 / (n . r)
    from sensorfault.stagea.geometry import pixel_rays_world

    eid = ent.pixel_entity[v["frame"]]
    rays = pixel_rays_world(K, v["R"], *v["depth"].shape)
    nr = np.where(eid >= 0, np.sum(ent.n[np.maximum(eid, 0)] * rays, -1), np.nan)
    rec = rc.frame_records(K, v, v["depth"] + 1.0 / nr, ent)
    np.testing.assert_allclose(rec.eps[:, 0], 1.0, atol=1e-6)
    np.testing.assert_allclose(rec.eps[:, 1:], 0.0, atol=1e-6)


def test_label_free_alignment_recovers_scale(block):
    K, views, _ = block
    rng = np.random.default_rng(2)
    s = 1 + 0.1 * rng.normal(size=len(views))
    raw = [v["depth"] / s[i] / 60.0 for i, v in enumerate(views)]
    res = alg.align_block(K, views, raw, "metric", fit_beta=False)
    assert res.history[-1] < 1e-3
    # the common factor comes from parallax only and is the least determined part (a few
    # percent on this small block); per-frame scales relative to it are recovered to 1%
    est = res.params[:, 0] / 60.0
    ratio = est / s
    assert abs(np.median(ratio) - 1) < 0.06
    assert np.max(np.abs(ratio / np.median(ratio) - 1)) < 0.01
    q = np.median(ratio)
    rel = [np.nanmedian(np.abs(d / q - v["depth"]) / v["depth"]) for d, v in zip(res.depth, views)]
    assert np.median(rel) < 0.01


def test_label_free_alignment_oblique_large_errors():
    # block-3-like: four 30 deg oblique headings per centre, per-frame scale errors of ~30%
    K, views, _ = synth.make_oblique_block(nx=4, ny=3)
    rng = np.random.default_rng(1)
    s = np.exp(0.3 * rng.normal(size=len(views)))
    raw = [v["depth"] / s[i] / 70.0 for i, v in enumerate(views)]
    res = alg.align_block(K, views, raw, "metric", fit_beta=False)
    assert res.history[-1] < 1e-3
    ratio = res.params[:, 0] / 70.0 / s
    assert abs(np.median(ratio) - 1) < 0.06
    assert np.max(np.abs(ratio / np.median(ratio) - 1)) < 0.005


def test_rho_and_permutation_null():
    rng = np.random.default_rng(3)
    M, F = 400, 8
    common = rng.normal(size=(M, 3)) * np.sqrt(0.4)
    ent = np.repeat(np.arange(M), F)
    frame = np.tile(np.arange(F), M)
    eps = common[ent] + rng.normal(size=(M * F, 3)) * np.sqrt(0.6)
    rec = rc.Records(
        ent, frame, np.tile(np.eye(3), (M * F, 1, 1)), eps, np.full(M * F, 9), np.zeros(M * F), M
    )
    info = {f: {"strip_id": 0, "strip_pos": f, "yaw": 0.0} for f in range(F)}
    ps = rp.pair_sums(rec, info)
    assert rp.rho_hat(ps, 0) == pytest.approx(0.4, abs=0.05)
    null = rp.pair_sums(rec, info, rng=np.random.default_rng(4))
    assert abs(rp.rho_hat(null, 0)) < 0.05
    L, e, n = rc.fuse(rec, 1.0)
    terms = rp.law_terms(rec, e, n)
    out = rp.law_ratio(terms, rp.rho_hat(ps, 0), np.arange(M))
    assert out["ratio"] == pytest.approx(1.0, abs=0.12)
    assert out["R_nominal"] == pytest.approx(1 / F)


def test_consensus_gate_flags_shifted_frame(block):
    K, views, ent = block
    rng = np.random.default_rng(5)
    preds = []
    for k, v in enumerate(views):
        d = v["depth"] * (1 + 0.005 * rng.normal(size=v["depth"].shape))
        if k == 3:
            d = d * 1.03
        preds.append(d)
    rec = rc.concat([rc.frame_records(K, v, p, ent) for v, p in zip(views, preds)])
    info = {v["frame"]: v for v in views}
    g = cg.gate(cg.frame_statistics(rec, info, 0.005, B=1))
    assert views[3]["frame"] in g["flagged"]
