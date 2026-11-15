# Pose graph + Gaussian field over the MDE depth field: what can be estimated, and what it gives the gate

Design note, 2026-11-14. Private. It extends design v3 (`src/sensorfault/eval/gaussian_uncertainty.py`,
committed in 4e90e48) and does not change the manuscript. Hypothesis and config names follow plan v3. Notation follows that module:
surfel k has a rendering shape (μ, R = [t₁ t₂ n], s, α) and a 9×9 information Λ_k over
θ = [δμ, δφ, δlog s]; the gate renders the (ν, τ₁, τ₂) normal-offset and tilt sub-block. View i has MDE
depth d̂_i, alignment (a_i, b_i), pose T_i, and noise Σ_ε,i = D_i C(ℓ, ν_nug) D_i. Numbers marked
*(est.)* are back-of-envelope estimates, not measurements.

## Answer

**One rule decides almost everything:** *any parameter fitted with view j's own depth is a nuisance for
view j's test.* Its Jacobian columns join span[d, 1] in the projection M. That costs degrees of freedom
and makes every fault in that span invisible. So the real question for each quantity is not only
"is it identifiable?" but also "may view j's depth touch it?".

| Quantity | Identifiable? | From which loss or coupling | What it gives the gate | Verdict | Cost: 8 GB / 24 GB |
|---|---|---|---|---|---|
| Per-view scale and shift (a_i, b_i) | Yes, up to one global scale gauge. Fixed by the metric prior a ~ N(1, σ_a²) or by a_1 = 1. Degenerate when var(d) is small in the view (fronto-parallel wall) and confounded with t_z there. | Depth loss. Closed-form 2×2 weighted least squares per view per iteration (Schur out). | Little directly: the test already projects span[d, 1]. Indirectly a cleaner field, because the views are fused after alignment. A large spread of a_i is a scale monitor, but by v2.4 it is absorbed and harmless. | **Estimate.** Refit (a_j, b_j) against the leave-j-out field, as the projection already does. | Negligible on both |
| Poses T_i | Yes, up to the Sim(3) gauge (SE(3) if the metric prior fixes scale). From depth alone, only where the normals span 3D (point-to-plane ICP conditioning). Corridors and planes leave sliding directions. | Pose-graph factors from matches (independent of the MDE). Depth-render loss through gsplat `viewmats` gradients. Optional photometric loss on the appearance block. | **Large**, if done the right way: hold the tested pose at its pose-graph-only estimate, with marginal Σ_q in Σ_r. Then pose faults stay testable, and depth and pose faults can be told apart (see §2). If the depth term of view j refines T_j, the 6 J_q columns are projected out and **pose faults become invisible**. | **Estimate. The tested view's pose excludes its own depth term.** | PG solve on the CPU, seconds for N ≤ 10³. Pose gradients add < 5% to a render step *(est.)* |
| MDE noise σ(range, texture, grazing), ℓ, nugget | Shared per model: yes, with ≥ 3 views per region (three-cornered hat) and enough covariate spread. Per view: identifiable, but must not be learned (it absorbs faults). ℓ is confounded with the field's projected correlation. | Clean dev data with ETH3D laser GT (the plan's week-2 fit). Online: one temperature γ per scene scaling Σ_ε, fitted robustly and leave-view-out (§9, decided). | Calibrated Σ_r is what makes the threshold valid (H5 calibration). A per-view σ_j learned from view j's own data would hide its fault: T/(n−2) **is** that σ_j-inflation estimate, so it is the test, not a parameter. | **Calibrate the shape offline; fit one temperature per scene (§9).** | FFT variogram / REML, ~0.1 s per view at 0.25 MP *(est.)*, on both |
| MDE bias, deterministic and shared (start: distance term d → d + c₂d² only) | Yes, if the same surface is seen at different depths or angles across views. The field pins the truth. | A few global parameters in the depth loss, all views. | Lower nominal residual and smaller σ, so more power. Removes a systematic misspecification that would otherwise raise the false-alarm rate. | **Estimate, shared only. Distance term first; add grazing or texture only if ETH3D dev residuals show them (decided).** | Negligible |
| MDE bias, per view (warp field) | Partly. The covariate-driven part is predictable from other views; the idiosyncratic part only from view j. In the Gaussian model it is the same object as a prior over the correlated part of Σ_ε. | Revised in §8. Prior mean predicted leave-view-out (cross-fitted) from other views; idiosyncratic part marginalised with a prior covariance calibrated on dev. | **Revised (§8).** Only a flat prior fitted on view j's own residual is harmful (hard projection: smooth warps and edge-aligned regions vanish). A cross-fitted mean plus calibrated covariance is valid, and its power cost is computable per fault family. Under correlated noise it is small for sharp-edged faults. | **Not in this paper (moved to a separate paper; §8 is background).** Here the per-view part stays marginalised in Σ_ε. In the other paper: estimate the predictable part leave-view-out, marginalise the rest, no edge-awareness in the per-view part. | ~10 floats per surfel for a view-dependent bias attribute (§8.4) |
| MDE bias, common to all views (glass, sky, OOD for the model) | **No.** Multi-view consistency cannot see an error every view shares. | Only external GT. | None. This is the "model-level fault", undetectable by construction; assumption A4 fails. | **Report as a limitation.** Measure its size on dev GT. | — |
| Per-view fault states z_j (and per region) | Yes, under the minority assumption (< ½ of the views of a region are faulty, ≥ 3 views to test). | Switch variables on view j's depth factors, as in switchable constraints / max-mixtures. Gate = E-step, refit = M-step. Hard decisions = the gate; soft weights = a baseline (not in the v3 list). | Iterative FDE (exclude, refit, re-test), as in multi-fault ARAIM. Excluding j is a subtraction plus one Newton step. **Needs sequential-testing α control.** | **Estimate (it is the gate).** | ~0.1 s per tested view (render + backward + CG) *(est.)*; 100 views ≈ 10 s |
| Surface uncertainty Λ_k / P_k | Only the (ν, τ₁, τ₂) block is observable from depth. Tangent position (2), spin about n (1) and scales (3) are prior-only, apart from silhouette edges. | Gauss-Newton blocks Jᵀ R⁻¹ J from the rasterizer (computed, not estimated). Adding poses and alignment gives a low-rank shared-error term (§3). | This is Σ_surf. The gate renders exactly the observable block. The pose-graph coupling gives a principled, cheap partial fix for the parked overconfidence. | **Compute.** Add the low-rank pose/alignment term. Keep the `information_deflation` ablation. | 1M surfels: Λ 45 + η 9 floats ≈ 216 MB. Per-(k, j) 3×3 blocks: 720 MB at v̄ = 20 (fits 8 GB) vs full 9×9 at 3.6 GB (24 GB only) |
| Intrinsics | Yes (bundle adjustment through the pose graph). Use the dataset calibration and pass f to Depth Pro. | Pose graph. | Shrinks U5. Otherwise minor. | Fix to the calibration. Refine only as an ablation. | Negligible |
| Opacity / visibility | Yes, but view j can "carve" (make surfels transparent) or create floaters. | Depth loss only (2DGS opacity); no photometric gradient. | Wrong support masks if view j shaped them. | **Decided: formulated as a 10th coordinate (logit α) of the uncertainty block; implemented frozen after warm-up (Algorithm in §10).** | 10×10 block: +10 floats per surfel (+40 MB per 1M) |

**Net:** estimate the following in the joint optimisation:
- poses (from the pose graph);
- per-view (a, b);
- the shared MDE bias (distance term only);
- fault switches;
- one noise temperature per scene (§9).

A per-view bias field (a prior mean predicted from other views plus a calibrated prior covariance) is
the subject of a separate paper; §8 records the argument as background.

Do not let anything per view be fitted to that view's own depth residual with a flat prior, apart from
(a, b). Doing so blinds the gate.

**What this buys:**
1. testable pose faults, with depth-vs-pose isolation (the plan's future-work item becomes cheap);
2. a calibrated Σ_r with a low-rank shared-error term, which partly fixes the parked overconfidence;
3. iterative exclusion.

**Compute:** everything fits on the 8 GB RTX 2070 at ~1 MP and ~1–2M surfels, *provided* the MDE runs
offline (cached depth) and the leave-out contributions are recomputed by a backward pass rather than
stored. A 24 GB card buys full resolution, 5–10M surfels, stored 9×9 per-(k, j) blocks (exact
leave-out of every coordinate), and the MDE co-resident.

---

## 1. Joint objective

```
min over {T_i, a_i, b_i, β, field, z_i}
  Σ_i z_i · ½ ‖ d̂_i − a_i·render_i(field, T_i) − b_i − B_i β ‖²_{Σ_ε,i⁻¹}      (depth, gated)
+ Σ_(i,i') ½ ‖ log(T_i⁻¹ T_i' Ẑ_ii'⁻¹) ‖²_{Ω_ii'}                              (pose graph, matches)
+ priors: a_i ~ N(1, σ_a²), field prior Λ_0, β ~ N(0, Σ_β), gauge on T_1
[+ appearance block: photometric loss on colours/SH; geometry gradients stopped, see §5]
```

Here β holds the shared MDE-bias coefficients and B_i their basis images (the distance term first;
grazing angle and texture only if dev residuals show them). Σ_ε,i is fixed in shape from dev
calibration and scaled by one scene temperature γ² (§9). z_i ∈ {0, 1} is the gate's decision. Mapping
is acausal (offline batch): every view, earlier or later, serves as a reference for every other.

Optimise by alternating:
1. the field by Adam on the depth loss (2DGS in gsplat), with (a, b) in closed form;
2. the poses by Gauss-Newton on the pose graph plus the depth-render term (the tested view excepted);
3. β by linear least squares;
4. Λ_k by accumulating the Gauss-Newton blocks at the current MAP (Laplace).

### Leave-view-out after optimisation

Once the field is optimised rather than fused in closed form, subtracting Λ_j alone is not enough. The
mean also moves. At the joint optimum ∇L_{−j}(μ) = −∇L_j(μ), so one Newton step on L_{−j} gives

```
Λ_{−j} = Λ − Λ_j,        μ_{−j} ≈ μ + Λ_{−j}⁻¹ ∇_μ L_j(μ)     (exact for the linear-Gaussian case)
```

This is the infinitesimal jackknife (influence function). Both Λ_j and ∇L_j come from **one backward
pass of view j**, so nothing per-(k, j) has to be stored. That is what makes it fit on 8 GB. Surfels
with Λ_{−j} ≈ Λ_0 (supported only by view j) get prior-sized variance and drop out of the support mask.
This is the optimised-field form of "the tested view spawns no primitives". Densification should still
not be triggered by a single view's gradients; require support from ≥ 2 views.

## 2. Poses: why the tested pose must exclude its own depth

The residual of view j is r_j = φ + J_q ξ + (noise), with J_q the 6 columns from the plan
(∂r/∂δt = −νᵀ/(ν·ℓ), ∂r/∂δω = −r(ℓ×ν)ᵀ/(ν·ℓ)), built per pixel from the rendered depth and normal.

**(A) T_j refined with view j's depth.** The best-fit ξ absorbs the projection of φ onto span(J_q).
The test must then use M with nuisance basis [d, 1, J_q]: dof n − 8. Pose faults (φ ∈ span J_q) have
κ = 0 and are undetectable. Depth faults lose their J_q component.

**(B) T_j from the pose graph without view j's depth term, with Σ_q in Σ_r.** This is the plan's U4.
Pose faults stay detectable whenever ξ_fault is large compared with Σ_q.

Use (B). It also gives **isolation at no extra cost**. Split the whitened residual into its component
in span(Σ_r^{−½} J_q) and the orthogonal complement:
- a pose fault concentrates in the first part (6 dof);
- a depth fault concentrates mostly in the second.

The pose graph's own residual on view j's edges is a second, MDE-independent channel:
- pose fault: high in both channels;
- depth fault: high in the depth channel only.

Per-view 6×6 depth information J_qᵀ Σ_r⁻¹ J_q, with its eigenvalues, should be logged as the degeneracy
check: on planes or corridors its small eigenvalues show which pose directions depth cannot test.

Gauge and confounding:
- Monocular poses fix scale only through the MDE metric prior and the a_i prior. Use log-mean a = 0 or a
  tight prior, or else the joint scale drifts between a and the field.
- On a fronto-parallel plane, b_j and t_z,j both shift the depth uniformly. Under (B) the pose graph
  pins t_z. Under (A) they are confounded, which is one more reason for (B).

**Colour.** Matches (and any photometric pose refinement) use colour. Design item 1 ("colour reaches
geometry only through the MDE") should be read as "the *map* geometry". Colour may reach the *poses* as
an independent channel, and that independence is exactly what makes (B) useful. **Decided:** colour may
refine poses (matches plus photometric refinement, batch over all frames). Photometric gradients into
surfel geometry and opacity stay stopped. One caveat: photometric pose refinement uses the appearance
block, which renders through the frozen geometry. A depth fault in view j therefore reaches T_j only
through the shared geometry, and the leave-view-out geometry is not what the appearance block saw.
This is second order; it is checked by the (A)/(B) ablation in §6.

## 3. Surface uncertainty and the parked correlation item

Per-surfel independent blocks ignore three things:
1. shared per-view pose error;
2. shared per-view alignment error;
3. within-view MDE correlation.

The joint model handles the first two exactly in low rank. Let θ_v = {T_i, a_i, b_i} (8N parameters)
with marginal Σ_θ from the pose graph and alignment. The surfel means depend on θ_v through the
back-projection (U = ∂μ/∂θ_v, sparse). The variance rendered into view j is then

```
σ²(u) = Σ_k w_uk² · g_ukᵀ P_k^{(ν,τ)} g_uk  +  (J_render,j U) Σ_θ (J_render,j U)ᵀ     (diag + low rank)
```

Keep the top r ≈ 20–50 eigendirections of Σ_θ. Each is one forward-mode (JVP) render of view j: r × ~5
ms on the 2070 *(est.)*. Σ_r then stays "stationary kernel + diagonal + low rank", and the plan's
Woodbury + FFT/CG still applies.

The third item (within-view correlation) does not reduce to low rank. Keep the existing
`information_deflation` (Λ_j × n_eff/n) as the conservative fix. It corrects the size of the variance,
not its structure, and is an ablation.

Observability: only (ν, τ₁, τ₂) carries depth information. The other six coordinates of the 9×9 are
prior-dominated. They are harmless to the gate, which renders only that block, but must not be reported
as learned surface uncertainty. Calibration check: leave-view-out whitened residuals on clean views
give T/(n−2) ≈ 1, before and after the scene temperature of §9.

## 4. Fault states inside the optimisation

1. Gate every view leave-view-out.
2. Exclude the most significant rejected view (subtraction + Newton step).
3. Re-test the remaining views against the cleaner reference, and stop when none are rejected.

This is multi-fault FDE. Control the family-wise false-alarm rate across iterations: use a
Bonferroni/Holm schedule, or calibrate the whole procedure by clean-run permutation, as the plan does
for single tests. Soft per-view weights from the same likelihood are a baseline, not the method.

Failure mode: if the faulty views are a majority in a region, the iteration locks onto the wrong
consensus. This already exists as `breakdown_faulty_views_dominate_region`. Region-level z (per
segment or per tile) uses the same machinery, with the scan statistic's regions.

## 5. GPU budget *(est.)*

Per surfel (structure-only, no SH):

| Item | Floats |
|---|---|
| Parameters (μ 3, quat 4, s 3, α 1) | 11 |
| Gradients | 11 |
| Adam state | 22 |
| Λ_k (45) + η (9) | 54 |
| **Total** | **≈ 98 (~400 B)** |

So 1M surfels ≈ 0.4 GB. For comparison, colour 3DGS spends ~48 extra floats per Gaussian on SH (×4 with
gradients and Adam ≈ 0.77 GB per 1M) if the appearance block is enabled.

| Component | RTX 2070 8 GB | 24 GB |
|---|---|---|
| MDE (Depth Pro, ViT-L based, FP16, batch 1) | Run **offline**, cache depth to disk, free it. Fits alone, not next to training *(unverified)* | Can be co-resident |
| Field, 1M surfels, structure only | ~0.4 GB | ~0.4 GB; 5–10M surfels feasible |
| Rasterizer workspace, ~1 MP view (gsplat packed) | ~0.5–1.5 GB | Same; full ETH3D resolution feasible |
| Leave-out data | Recompute per tested view (1 backward pass) | Store per-(k, j) 9×9: 3.6 GB at 1M × v̄=20 |
| Low-rank pose term | r JVP renders per tested view | Same, larger r |
| Σ_r whitening (FFT/CG) | ~50 CG iterations at 0.25 MP, ≪ 1 GB | — |
| Pose graph (GTSAM-class, CPU) | Seconds for N ≤ 10³ | — |
| Training time (structure + depth, 1M surfels, ~10–30k iterations) | Tens of minutes per scene *(est.)* | ~2–3× faster |

Dense cross-surfel covariance (9M × 9M) is impossible on any GPU. Only the low-rank shared-error form
in §3 is tractable.

## 6. Evaluation hooks (no new hypotheses implied)

- Ablation: tested pose (A) refined with own depth vs (B) pose graph only. It needs injected pose
  faults, which plan v3 (4e90e48) dropped from scope, so new configs would be required. Expected: (A)
  loses pose-fault power and keeps depth-fault power minus the J_q component.
- (Separate paper, not this one.) Per-view bias field as (i) flat prior fitted on own view, (ii)
  calibrated prior with zero mean, (iii) calibrated prior with cross-fitted mean, (iv) none. Fault
  families: region, smooth warp, edge-aligned region. Expected: (i) hides warps and edge-aligned
  regions; (iii) gives the lowest false-alarm rate at the smallest MDB loss.
- Ablation (§9): raw vs temperature-scaled thresholds, false-alarm rate per scene and per view.
- Ablation: low-rank pose/alignment term in Σ_surf on/off, and `information_deflation` on/off. Metric:
  clean T/(n−2) and false-alarm rate (H5).
- Baselines unchanged (SpotLessSplats, RobustNeRF, robust depth losses, per-view aligned RMS, MDE
  confidence). They are run on the same joint optimisation with z fixed to 1.

## 7. Decisions (Murat, 2026-11-14)

1. Colour may refine poses; mapping is acausal (offline batch). §2.
2. Opacity: formulated as the 10th coordinate, implemented frozen after warm-up. §10.
3. Shared bias: distance term only; add grazing or texture only if ETH3D dev residuals show them.
4. Noise: σ(·), ℓ, ν calibrated offline, plus one free temperature per scene, fitted robustly and
   leave-view-out. §9. Follow-up: the energy threshold must also absorb the per-view heterogeneity ω
   measured on dev, which is a new week-2 number (§9, item 3).
5. Per-view bias field: split into its own separate paper. In this paper the bias stays the shared
   distance term only. §8 stands as background, not as the main method. The size of Depth Pro's
   idiosyncratic distortion on ETH3D dev (week-2 calibration) is still worth recording, because it
   feeds both papers.

## 8. Per-view bias field, revisited (background; a separate paper)

> **Status (Murat, 2026-11-14):** this is not part of this paper's method. The per-view bias field is
> split into a separate paper; here the bias is the shared distance term only (§7, item 3). This
> section records the argument and the numbers as background for that paper, and to justify why this
> paper keeps the per-view part marginalised in Σ_ε.

The question: a semi-self-supervised diffusion over the structure-only, information-form field fits a
per-view bias field leave-view-out, so view j's own depth never sets view j's correction. Can a field
flexible enough to fit real MDE distortions be kept from absorbing a real fault?

**Short answer.** Partly yes, and my earlier rule was too strong.
- The rule is only right for a **flat prior fitted on view j's own residual**. That is a hard
  projection, and it does blind the gate (numbers in §8.3).
- A field whose **mean is predicted from other views (cross-fitted)** and whose **remaining
  idiosyncratic part is marginalised with a prior covariance calibrated on dev GT** keeps the null
  valid. Its power cost is computable in closed form for each fault family. Under realistic correlated
  noise that cost is small for sharp-edged region faults.
- What no field can do is separate an idiosyncratic distortion from a fault of the same shape and size.
  By definition such a fault is nominal. The definition we commit to: *a fault is what is improbable
  under the calibrated nominal model, distortion prior included.*
- How much flexibility is affordable is then an empirical question: how large is Depth Pro's
  unpredictable distortion on ETH3D dev?

### 8.1 Estimating and marginalising are the same test

For residual r (after alignment) and a field Bβ with Gaussian prior β ~ N(m, Σ_β):

```
min_β ‖r − Bβ‖²_{Σ⁻¹} + ‖β − m‖²_{Σ_β⁻¹}  =  (r − Bm)ᵀ (Σ + B Σ_β Bᵀ)⁻¹ (r − Bm)
```

So a fault never silently "disappears" into β: whatever β absorbs, the prior term charges back. The
test, its null and its detectability κ² = φᵀMφ depend on only three things:
1. the prior mean m (what other views predict);
2. the prior covariance Σ_β (how much unpredictable distortion counts as nominal);
3. the structure B (which patterns are cheap to explain).

A flat prior (Σ_β → ∞) moves B into the projection, so κ = 0 for every φ in span B. Fitting the field
with view j's own residual and then "testing what is left" is this flat case, unless the prior penalty
is also added to the statistic.

### 8.2 What a leave-view-out field can and cannot predict

Decompose view j's distortion as β_j(u) = β_cov(c_j(u)) + β_scene(x(u)) + β_idio,j(u).

| Part | What it is | Predictable from other views? | Role |
|---|---|---|---|
| β_cov | A function of view-dependent covariates c: distance, grazing angle, image position, local image content | **Yes.** The same function acts in every view at different covariate values. | The prior mean m_j. It generalises the shared distance term. |
| β_scene | The same error at the same 3D point from every view (glass, sky) | Not needed and not visible: it moves the fused field itself | Model-level fault. Invisible to field and gate alike. |
| β_idio,j | Specific to this image (the network's response to this crop) | **No.** Only view j's own data sees it. | Marginalised with calibrated Σ_β. Here real distortion and real fault compete, and only prior probability separates them. |

One consequence for the "diffusion over the Gaussian field" reading. A bias field anchored to 3D
positions and diffused over the surfel graph carries **no** information about view j. At the MAP, the
per-surfel average residual of the other views is zero by construction, because the field fits them.
What transfers is the map from *view-dependent* covariates to distortion. So the field should be a
**view-dependent attribute** of each surfel, the analogue of SH colour:

```
β_k(view) = b_k0 + b_k1·log(distance) + low-order SH(direction in the surfel frame)      (~4–9 coeffs)
```

It is fitted per surfel from the other views' leave-out residuals and evaluated at view j's distance
and direction. The result is rendered into view j as m_j, and its predictive variance is added to
Σ_β. It is identifiable where the views of surfel k span distances and directions, which is the same
condition as for the shared distance term, now local. "Semi-self-supervised": the labels are other
views' residuals (self-supervision), and dev GT calibrates Σ_β (supervision).

A *generative* diffusion model as a prior for β_idio (trained on dev distortion maps) is the other
reading. It gives a non-Gaussian Σ_β: no closed-form κ or MDB, and calibration by permutation only.
Parked.

### 8.3 Numerical check of the power cost

Setup (toy; `tests/core/test_bias_power.py`):
- 32×32 view (n = 1024), depth ramp 2–4, nuisance [d, 1] as in the gate;
- matched-statistic MDB = 3.24/κ in units of σ;
- fault families: a sharp-edged square region (f = 0.12) and a smooth zero-mean warp (rms 1);
- field B: (a) smooth DCT basis k×k; (b) edge-aware piecewise affine in d on 5 segments, one of which
  either coincides with the fault region or misses it by 4 px;
- prior on the field: flat, or Gaussian with field rms 0.5σ or 0.25σ. The 0.25σ case stands for "other
  views predict 75% of the distortion variance".

MDB (σ units); lower is better, ∞ = undetectable.

| Field model | iid noise: region | iid noise: warp | correlated ℓ=3, nugget 0.2: region | correlated: warp |
|---|---|---|---|---|
| none ([d, 1] only, current gate) | 0.317 | 0.101 | 0.408 | 0.486 |
| smooth 3×3, flat | 0.424 | ∞ | 0.413 | ∞ |
| smooth 6×6, flat | 0.691 | ∞ | 0.418 | ∞ |
| smooth 10×10, flat | 0.870 | ∞ | 0.431 | ∞ |
| smooth 6×6, prior rms 0.5 | 0.573 | 0.292 | 0.411 | 0.570 |
| smooth 6×6, prior rms 0.25 | 0.453 | 0.170 | 0.409 | 0.509 |
| smooth 3×3, prior rms 0.5 | 0.406 | 0.582 | 0.410 | 0.751 |
| edge-aware, segment = fault, flat | ∞ | 0.393 | ∞ | 0.581 |
| edge-aware, segment = fault, prior rms 0.5 | 1.612 | 0.113 | 1.628 | 0.490 |
| edge-aware, segment misses by 4 px, flat | 0.373 | 0.394 | 0.412 | 0.582 |
| edge-aware, segment misses, prior rms 0.5 | 0.348 | 0.112 | 0.411 | 0.490 |

What it shows:
1. **Under correlated noise, a smooth per-view field is almost free for sharp-edged faults.** The
   region MDB goes from 0.408 to 0.41–0.43σ even with a flat prior. The correlated-noise model already
   discounts smooth patterns, so the evidence sits in the edge, which a smooth field cannot imitate.
   Under iid noise the same field costs up to 2.7× (0.32 → 0.87σ). The cost must be computed with the
   calibrated Σ_ε, never guessed from an iid picture.
2. **Smooth warps are what a flexible smooth field takes away.** With a flat prior they vanish. With a
   calibrated prior the MDB rises 5–55% (0.49 → 0.51–0.75σ, correlated). The cross-fitted mean is what
   pays: shrinking the unpredictable rms from 0.5 to 0.25 recovers most of the loss.
3. **Edge-awareness in the per-view part is the dangerous choice.** When a segment coincides with the
   fault (the glass-window case), a flat prior makes it undetectable, and even a calibrated prior
   quadruples the MDB (0.41 → 1.63σ). A segment 4 px off costs almost nothing. Real glass and sky faults
   follow image edges, so an edge-aware per-view field is tuned to absorb exactly the faults the paper
   is about.

Caveats: a toy grid; matched statistic only (energy-test costs differ, being smaller per pattern but
spread over all patterns); the warp is one DCT basis function (worst case for the flat smooth field);
one fixed noise ℓ. The 32×32 script that produced this table, overwritten in the scratchpad, has been
recreated as `tests/core/test_bias_power.py`. It regenerates every cell above to ±0.0015σ, checks the
iid baselines against the closed forms (region ≈ 3.24/√(n f(1−f)); warp = 3.24/√n), and asserts
points 1–3. It runs in about 1 s on the CPU.

**Independent cross-check (64×64, n = 4096).** This was a different construction, whose script
survives as
`/tmp/claude-1000/-home-eiki/5227ee46-e127-5923-accd-c6b8c9bf0265/scratchpad/bias_power_64.py`
(scratchpad only, so promote it to `tests/core/` before relying on it):
- scene: a sloped background and a box at a nearer depth, giving three segments;
- correlated noise ℓ = 4, nugget 0.2;
- smooth basis: cosine k×k;
- the "prior rms τ" is the field's total rms in σ.

MDB in σ, with the ratio to the current gate in brackets.

| Field model | Region on the box (edge-aligned, f = 0.12) | Region inside the wall (no edge, f = 0.12) | Smooth warp (rms 1) |
|---|---|---|---|
| [d, 1] only (current gate) | 1.10 | 0.254 | 0.371 |
| smooth 4×4, flat | 1.82 [1.64] | 0.257 [1.01] | 2.34 [6.3] |
| smooth 8×8, flat | 1.83 [1.65] | 0.259 [1.02] | 3.30 [8.9] |
| smooth 8×8, prior τ = 0.3 | 1.15 [1.04] | 0.255 [1.00] | 0.394 [1.06] |
| smooth 8×8, prior τ = 1.0 | 1.38 [1.25] | 0.257 [1.01] | 0.556 [1.50] |
| edge-aware affine per segment, flat | ∞ | 0.267 [1.05] | 0.371 [1.00] |
| edge-aware, prior τ = 0.3 | 1.72 [1.56] | 0.265 [1.04] | 0.371 [1.00] |

This confirms points 1–3:
- flat smooth fields kill warps (×6–9);
- calibrated priors cost ≤ 6% at τ = 0.3σ;
- an edge-aware field fully absorbs the edge-aligned region under a flat prior, and still costs ×1.56
  with a prior.

It also shows **a new effect, independent of any bias field.** A region fault on an *object at a
distinct depth* is already partly absorbed by the gate's own [d, 1] alignment: MDB 1.10σ, against
0.254σ for an equal-size region on the wall. The reason is that the object's indicator is nearly
affine in d. Glass and sky faults are often exactly such depth-distinct regions. The regime map (H1)
should therefore vary the region's depth contrast, not only f, and the matched scan should report
κ² after projecting [d, 1] per candidate region.

Under iid noise (same script) the costs are much larger: edge-aligned region ×7.8–10 for flat smooth
fields, and warps ×11–31. This is the reason, again, to compute costs only with the calibrated Σ_ε.

### 8.4 Regularity that keeps it identifiable, and what each costs

- **Prior mean from other views (cross-fitting).** This is the main lever.
  - The residuals that train the predictor must be computed against fields that also exclude view j.
    Otherwise j's fault leaks into its neighbours' residuals with opposite sign, scaled by j's fusion
    weight, and the predictor learns a fraction of −φ_j.
  - Use K-fold cross-fitting (K = 5), as in double/debiased ML (Chernozhukov et al., Econometrics J.
    2018, from memory). For fold k: subtract fold k's information once, then for each training view
    i ∉ k take one more subtraction plus a Newton step to get r_i against field_{−k,−i}; fit the
    surfel attributes; evaluate them for the views in fold k.
  - Out-of-fold prediction error goes into Σ_β.
  - Cost: K extra field states (information subtraction, no retraining) and K attribute fits. About
    40 MB per 1M surfels per fold for the attributes *(est.)*.
- **Smoothness.** Allowed in the idiosyncratic part, with Σ_β calibrated on dev. Cost: small for
  region and edge faults, real for smooth warps (point 2 above). Report the warp MDB with and without
  it.
- **Edge-awareness.** Allowed only inside the predicted mean m_j, which is fitted on other views and so
  cannot absorb view j's fault. Not in the idiosyncratic per-view part (point 3 above). A global
  edge-aware mean that "corrects" a glass error seen the same way in every view is correcting a
  scene-anchored error, which the gate could not see anyway.
- **Rank.** Keep p ≪ n_eff. For every field model, report κ² with and without B per fault family. That
  is the price of the nuisance, in closed form, per view.
- **Amplitude.** Σ_β comes from dev GT, after the shared distance term and the cross-fitted mean are
  removed. If the remaining idiosyncratic rms is below about 0.25σ, the field is cheap; if it is near σ
  or above, the field dominates Σ_r and the sub-noise argument weakens. That is a result to report, not
  to tune away. It goes with the week-2 go/no-go.

### 8.5 Position against AffineGS (arXiv 2611.07709; Tse, Peng, Chen, Yao; arXiv comments say NeurIPS 2026)

Read from the arXiv HTML v1. Main text verbatim; appendices only as a summary, so the appendix numbers
here are unconfirmed.

**What they do:**
- A dense per-pixel affine field in disparity space, 1/d = α(u)D(u) + β(u). D is the rendered 3DGS
  depth, and d is the Depth-Anything-V2 disparity normalised to the render's range.
- α = ‖∇d‖/(d²‖∇D‖ + ε) and β = 1/d − αD are re-estimated every iteration from **the view's own MDE
  and its own current render**.
- Percentile clamping, then a guided filter with the disparity as guide and a very wide window (radius
  ~550–650 px on MuSHRoom). The filter is the edge-awareness; there is no explicit smoothness
  regulariser.
- Gradient and normal-consistency losses tie the rectified depth to the rendered depth and normals. A
  global a, b least-squares fit anchors scale.
- No uncertainty, no confidence weighting, no held-out fitting, no fault or consistency test.
- Dense scenes are initialised from sensor depth (following DN-Splatter).
- Their Table 1: one global affine gives ~59% inliers vs ~92% with K = 5 regions (MuSHRoom,
  ScanNet++). How the regions are chosen is not stated, so this may be an oracle analysis.
- Results: better depth and mesh than DN-Splatter (monodepth) on MuSHRoom (F 0.71 vs 0.28). Not the
  best F on ScanNet++ (0.32 vs 0.42–0.47 for 2DGS, SuGaR, Nerfacto); the paper says so.

**In our terms, AffineGS is the configuration in §8.3 with the worst power:** an edge-aware, flat-prior
nuisance fitted with the view's own data. For geometry that is a sensible kind of robustness: a faulty
region is rectified toward the render, so it adds little harm where other views support the render.
But it makes no statement:
- there is no detection, false-alarm rate, MDB or protection level;
- where the faulty view is the main support of the render, the self-referential fit locks the fault
  in;
- an edge-aligned region fault is exactly what its field is built to absorb.

| | AffineGS | Here |
|---|---|---|
| Goal | Correct depth to improve geometry | Treat distortion as a nuisance inside a calibrated test |
| Who sets view j's correction | View j's own MDE and render, every iteration | Other views (cross-fitted mean) plus a dev-calibrated prior for the rest |
| Edge-awareness | Guided filter on the view's own disparity | Only in the cross-fitted mean |
| Uncertainty | None | Σ_β and predictor error in Σ_r; κ, MDB, PL in closed form |
| Output | Rectified depth, better mesh | Per-view decision with α, MDB, protection level; fused field from kept views |
| Depth model | Relative disparity (DA-V2) | Metric depth (Depth Pro); its distortions may be smaller and differently shaped |

How to use it:
1. **As a baseline** (in the separate paper; optional here): AffineGS-style rectification, then
   per-view aligned RMS as the test. Plan v3 has no estimate-and-correct registry slot any more, so
   this would be a new stub next to `configs/baselines/cheap_view_tests.yaml`.
2. **As evidence** that a global affine is inadequate for relative-depth models. We must measure the
   same for Depth Pro on ETH3D dev before assuming it.
3. **As a candidate structure** (piecewise affine on disparity edges) for the *cross-fitted mean* only.

The claim of difference: "they correct depth; we treat the correction as a nuisance with a calibrated
prior inside a leave-view-out test, and we show its power cost". To be re-checked against their
appendices and code when available.

## 9. Noise temperature (decided)

Σ_ε,i = γ²·D_i C(ℓ, ν) D_i. The shape (σ(·), ℓ, ν) comes from dev GT; one scalar γ per scene is fitted
on the scene itself. This is the analogue of temperature scaling (Guo et al., ICML 2017).

**Estimator (pre-registered before the test scenes).** For the test of view j:

```
γ̂²_{−j} solves  median_{i≠j} [ T_i(γ) / q₀.₅(χ²_{m_i}) ] = 1
```

T_i is view i's leave-out energy statistic, m_i its test dof (n_i − 2, or the Satterthwaite dof from
n_eff under correlation), and q₀.₅ the χ² median. T_i(γ) is monotone in γ, so this is a 1-D root.
With Σ_ε dominating Σ_r, γ̂² is just the median ratio. Threshold view j against
F(m_j, ν_pool) ≈ χ²_{m_j}/m_j, because γ̂_{−j} does not use view j.

**What it does to the false-alarm guarantee:**
1. **Scale-only misspecification is fixed.** If the shape is right and only the level transfers badly
   from dev to the scene, α is restored. The finite-sample error is small: the relative sd of γ̂² is
   about 1.25·√(2/n_eff)/√(N−1) (median efficiency), or 0.8% at n_eff = 10³ and N = 30.
2. **Faults make it conservative, not anti-conservative.** Faulty views have larger T_i, so they push
   the median up, raising γ̂ and lowering both the realised α and the power. The size is small: with a
   fraction ε of faulty views (< ½), the median moves to the 1/(2(1−ε)) quantile of the clean
   distribution. For ε = 0.2 that is the 0.625 quantile, so γ̂² rises by ~0.32·√(2/n_eff) ≈ 1.4% at
   n_eff = 10³. Leakage of view j's fault into its neighbours' T_i (their fields include j) also
   inflates γ̂: conservative again. Breakdown at ε = ½, consistent with the minority assumption.
3. **What it cannot fix:**
   - wrong shape (ℓ, ν, the σ(·) pattern);
   - heavy tails;
   - per-view heterogeneity of the nominal noise.

   The guarantee becomes a **scene-average** false-alarm rate (averaged over views ≈ α), not a per-view
   conditional α. A view whose nominal noise is genuinely larger (out-of-distribution content) keeps an
   elevated false-alarm rate. It is indistinguishable from a fault by design (§8's definition). Fitting
   it away would need a per-view temperature, which is the fault-absorbing parameter refused in the
   table.
   **How fragile the energy test is to heterogeneity (computed).** The energy statistic has m ≈ n_eff
   or more dof. That makes it as much a test of the noise *level* as of faults. Let r be view j's true
   noise scale relative to γ̂. The per-view false-alarm rate at nominal α = 0.05 is then:

   | r | m = 10³ | m = 10⁴ |
   |---|---|---|
   | 1.00 | 0.050 | 0.050 |
   | 1.01 | 0.117 | 0.408 |
   | 1.02 | 0.229 | 0.877 |
   | 1.05 | 0.710 | 1.000 |
   | 1.10 | 0.995 | 1.000 |

   The matched region scan (R = 100, Bonferroni) only scales z by 1/r. Its family false-alarm rate goes
   0.050 → 0.064 → 0.092 → 0.155 at r = 1.00, 1.02, 1.05, 1.10.

   Consequences:
   - One scene temperature is necessary but not sufficient for the energy test.
   - Its threshold must absorb the per-view heterogeneity ω (sd of log r across clean views), measured
     on dev after temperature scaling. Holding α at m = 10³ gives c/m = 1.075, 1.102, 1.196 and 1.400
     for ω = 0, 0.02, 0.05 and 0.10. The required noncentrality goes 139 → 191 → 376 → 819, so the
     energy MDB grows ×1.17, ×1.64 and ×2.43.
   - The scan test barely moves.
   - **ω is a week-2 go/no-go number next to σ(·), ℓ, ν.** If ω ≳ 0.05, lead with the scan statistic
     and keep the energy test for unknown patterns at its heterogeneity-calibrated threshold.
   - Pixel-count dof (m = n − 2 ≈ 10⁵) must not be used for the energy threshold. Use n_eff or the
     dev-calibrated null.

4. **Comparison with Guo et al.** As there, one parameter does not change the ranking of views within
   a scene (exactly when Σ_ε dominates Σ_r), so the ROC/AUC within a scene is unchanged and only the
   operating point moves. Unlike there, γ is fitted transductively on the test scene's unlabelled
   views. That is valid only under the minority-fault assumption and only because it is leave-view-out.
5. **It doubles as a scene-level monitor.** γ̂ far above its dev range (γ̂ > γ_max, fixed on dev) means
   the model is inadequate for the scene. Raise the scene-level alarm instead of gating, which ties in
   with `breakdown_misspecified_measurement_model`. Report γ̂ per scene, and H5 calibration both raw
   and temperature-scaled.

## 10. Opacity as the 10th coordinate, frozen after warm-up (decided)

**Formulation.** θ_k = [δμ (3), δφ (3), δlog s (3), δ logit α (1)], with a 10×10 information block.
Depth sees opacity through the compositing weights: ∂D(u)/∂logit α_k = α_k(1−α_k)·∂D(u)/∂α_k, which
includes the transmittance of the surfels behind. The rasterizer backward pass provides it. The
(ν, τ₁, τ₂) gate block is unchanged.

**Implementation: Algorithm "train, freeze, fuse, test" (offline batch).**
0. *Precompute.* Depth Pro depth for every view, cached to disk (the MDE is not resident during
   training). σ(·), ℓ, ν from dev. Poses from matches plus photometric refinement over all frames
   (colour allowed, §2).
1. *Warm-up (iterations 0…K_w, K_w ≈ 3–7k, chosen on dev).* Optimise μ, R, s and logit α on the depth
   loss over all views.
   - (a_i, b_i) in closed form each iteration; the shared distance term by least squares every M
     iterations.
   - Densify only where ≥ 2 views support a split or clone. Prune at α < α_min.
   - **Opacity receives depth gradients only.** The appearance block (colour/SH) trains alongside,
     with stopped gradients into μ, R, s and α.
2. *Freeze at K_w.* Opacity is fixed, and densification and pruning stop. Continue μ, R, s and poses
   (the tested pose by rule (B), §2) to convergence. The appearance block continues on the frozen
   geometry.
3. *Laplace.* Accumulate Λ_k (10×10) and η_k at the MAP, together with Σ_θ for poses and alignment
   (§3).
4. *Gate, per view j.*
   - One backward pass of view j gives Λ_j and ∇L_j.
   - Newton leave-out on the 9 geometric coordinates. Opacity stays fixed: it is not re-estimated
     without j.
   - **Support rule for the 10th coordinate:** drop surfel k from view j's leave-out render if view j
     supplied more than a fraction ρ (e.g. ½) of its opacity information, i.e.
     Λ_j[α,α]/Λ[α,α] > ρ. Such a surfel's visibility was set mainly by the tested view, so its pixels
     count as untestable rather than as a reference.
   - Render d and σ² into j. Align, whiten with γ̂_{−j}, test. Iterative exclusion as in §4.
5. *Output.* Refit on the kept views; the fused field is the output.

**Residual risk.** Surfels that view j caused to be *pruned* during warm-up cannot be restored by any
leave-out. The support rule only handles surfels that still exist. The ablation for this is a fold-wise
warm-up (K = 5 warm-ups, each without one fold; 5× the warm-up cost, which fits the 24 GB card's time
budget better). It measures whether frozen opacity changes the false-alarm rate or the MDB.

Memory: 10×10 symmetric = 55 floats instead of 45, so +40 MB per 1M surfels. Per-(k, j) recomputation
stays a single backward pass.
