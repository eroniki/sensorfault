# View-level fault detection and exclusion for learned-depth reconstruction — research plan (design v2.5)

Target: IEEE Access, solo author, 14–18 evening weeks (~10 h/wk), one 8 GB consumer GPU (RTX 2070 on this machine).
Title (fixed): *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*.
Status: v2.5 of 2026-11-14. **v2.5 reframes the surface** (depth maps only; closed-form Gaussian planar primitives that separate extent from uncertainty; information-form fusion with exact leave-view-out = RAIM solution separation; ray-plane backward rendering and alpha compositing; projective association; per-region posterior, protection level and view count; no adaptive densification, fixed-budget MCMC only as an ablation). v2.4 derives the gate from an explicit measurement model** (nominal measurement with scale/offset ambiguity, heteroscedastic spatially-correlated noise, pose, intrinsics and surface uncertainty → residual covariance Σ_r → whitened, nuisance-projected tests; n_eff, MDB and protection level computed from Σ_r, not assumed). Framing = **integrity monitoring (RAIM-style FDE) on views**. Kept: one metric depth model per image (default Depth Pro), shared Gaussian-surface consensus (closed form, leave-view-out by subtraction, doubles as fused output), "gate is the contribution, pipeline is the test bed". Design v1 (physical-sensor faults) is in `configs/legacy_sensor_faults/`. Items marked **[UNVERIFIED]** come from memory or abstract-only reads.
Hygiene: public datasets, public pretrained models and public code only.

---

## 1. Framing, core argument and honest gap

**RAIM ↔ this paper.**

| RAIM / FDE | Here |
|---|---|
| redundant satellite ranges | other views of the same scene (a single model gives no redundancy within a view) |
| fault: a faulty satellite | view-level fault: depth bias/warp/failed region **or** pose fault (spoofed prior, glitch, false loop closure) |
| consistency test on residuals (range comparison, parity, residual, solution separation) | per-view test of the whitened, nuisance-projected residual against the rendered shared surface: energy T = r₀ᵀMr₀ ~ χ²(n−2) and a matched region scan |
| threshold from false-alarm probability | per-view false-alarm α (5%), threshold calibrated on clean dev data |
| minimum detectable bias for missed-detection β | β_min = (z_α+z_β)/κ, κ² = φᵀMφ (detectability of the fault pattern after whitening by Σ_r and projecting out scale/offset); iid pixels: κ² = n·f(1−f)·δ² |
| exclusion | g_i = 0 (drop the view's depth from PGO and fusion) |
| alert / loss of integrity | no-consensus alarm (unsupported view) and scene-level alarm (>½ views rejected) |
| availability | fraction of views tested and kept |
| protection level | conditional PL = max_i |s_i|·δ_min,i·σ + z_q·σ_q (single fault; assumptions A1–A4) |

**Core argument (corrected in v2.4).** A coherent view-level fault adds a small, *spatially structured* bias across a view (over part of it, a warp, or a pose error that moves the whole rendering) that **the monocular scale and offset cannot absorb**. At sub-noise per-pixel size (e.g. 0.3σ) a pixel filter (|ρ|>k=3) sees almost nothing: flagging probability 0.0040 vs 0.0027 null. Aggregated over the view's co-visible pixels, **whitened by Σ_r and projected orthogonally to span{[d,1]}**, the same pattern is a large statistic. **The gate wins exactly when faults are sub-noise per pixel but coherent across part of a view that the scale/offset cannot absorb** (glass, sky, out-of-distribution scenes). It should *not* win when a fault is large but confined to few pixels (pixel filters are the right tool) or for large false loop closures (switchable constraints / DCS / max-mixtures were built for them).

**Correction that follows from the measurement model.** With a free per-view offset b (and scale a), a bias that is **uniform over the whole view (f=1), or proportional to σ(d) when σ is affine in depth, lies in span([d,1])**: it is absorbed by the alignment in the gate *and* by the per-view alignment variables in the PGO. It is **not a fault, is not detected, and is not harmful** (this is the affine control, H1d). The earlier statement "a bias on every pixel of a view, f=1, is detected" was wrong. Detectability of a bias on a fraction f of independent pixels is κ² = n·f(1−f)·δ²: peaks at f=½, **vanishes at f=1**. Real failures (glass, sky regions; warps) are partial or non-affine, which is why the regime map uses f ≤ 0.8 and a separate f=1 control.

**Verified numbers (code: `src/sensorfault/eval/detection_theory.py`; tests in `tests/core/test_detection_theory.py` and `tests/core/test_measurement_model.py`; Monte Carlo cross-checked).**
- Pixel filter, δ=0.3, k=3: flag prob 0.00395 vs 0.0027 null; second-order excess kφ(k)δ².
- MDB (α=0.05, β=0.1, z_α+z_β=3.24), n=1000 independent pixels, bias on fraction f (offset nuisance removed):

| f | matched (region known) | scan, R=100 | energy (unknown pattern) |
|---|---|---|---|
| 0.05 | 0.47σ | 0.69σ | 1.71σ |
| 0.2 | 0.26σ | 0.38σ | 0.93σ |
| 0.5 | 0.21σ | 0.30σ | 0.75σ |
| 1 | ∞ (absorbed) | ∞ | ∞ |

  (the earlier table "0.10σ at f=1, n_eff=1000" assumed a free mean and is **withdrawn**.) Energy λ₀ = 138.8 at 998 dof (exact noncentral χ²; leading-order 130.7).
- Matched statistic vs aggregated pixel-count test at δ=0.3: ratio 12.4 (f=0.5), 24.9 (f=0.2); the count overtakes only above δ≈1.65σ (f=0.5) / 2.30σ (f=0.2).
- Energy statistic detects a zero-mean warp (rms 0.5σ, n=1000: power 0.998) that a mean statistic cannot (0.118) — and the signed mean is not used at all once the offset is fitted.
- n_eff = tr(C)²/tr(C²) (energy); mean statistic n_mean = n²/(1ᵀC1). Gaussian-kernel closed forms (with nugget ν): n_eff ≈ n/[(1−ν)²πℓ²+2ν(1−ν)+ν²] (grid check within ~9%); n=10⁵, ℓ=6, ν=0.2 → ≈1.4×10³.
- **A nugget is required**: without it Σ is numerically singular and any sharp-edged pattern looks infinitely detectable (κ≈4.8×10⁵ in a test). Sharp-edged region faults are *more* detectable than n_eff suggests (correlated noise cannot imitate an edge); smooth warps *less*.
- Pose Jacobian of rendered depth: ∂r/∂δt = −νᵀ/(ν·ℓ), ∂r/∂δω = −r(ℓ×ν)ᵀ/(ν·ℓ); verified against exact ray–plane intersection; rotation-row norm = r·tanθ, so the earlier pointing-error variance (rσ_φ tanθ)² is the diagonal special case of Σ_r.
- **The weak point is the measurement model itself** (Gaussianity, heavy tails, calibration of σ(·), ℓ, ν, noisy Σ_surf from small K).

**Why not robust kernels / soft down-weighting.** They down-weight when a residual is large (a sub-noise coherent bias passes through) and give no trust statement: no exclusion, no false-alarm/missed-detection probability, no alarm. Soft per-view weights inside the differentiable PGO are the closest relative and are a **baseline (H5)**; if they match the gate in the sub-noise regime the paper's premise fails — H5 is written to find out.

**Fault models.** Depth: coherent bias (δ, f — the regime map), smooth warp, region failure, non-affine distortion. Pose: spoofed pose prior (consistent/drifting offset on a block), intermittent glitches, false loop closures. A pose error shifts a whole view's residual through the same pointing-error term (r·δφ·tanθ), so one test covers both; *isolating* depth-vs-pose is future work.

**Test bed (not the framing).** RGB → one metric depth model per image (default **Depth Pro**: absolute-scale depth, no intrinsics needed, fast 2.25 MP in 0.3 s per its authors, sharp; alternatives UniDepth, Metric3D v2; chosen on dev scenes only) → matches → provisional poses → closed-form shared Gaussian surface (information fusion, leave-view-out by subtraction) → gate → **differentiable, renderable PGO** (optional render-based depth constraint; soft down-weighting = baseline) → surface refitted on kept depth = fused output. Ensemble of models = ablation only (H10).

**Baselines the gate must beat.** Robust kernels (Huber/Cauchy/GNC; switchable/DCS/max-mixture for loop closures), soft view weights, COLMAP consistency filter (+ the same flags aggregated into a count test = strongest pixel competitor), cross-view voting. **Reference points, not rivals:** MASt3R-SfM, VGGT, COLMAP SfM.

**Novelty (limited search, to be re-checked).** No work found applying RAIM-style FDE to the views of a learned-depth reconstruction or gating per view's depth before PGO. Related: integrity monitoring for LiDAR navigation (Joerger et al., Sensors 2018). UncLe-SLAM (arXiv 2306.11048) is single-stream pixel-wise uncertainty, not multi-source. Literature pass still to do: integrity monitoring for visual/learned navigation; depth-prior filtering in SfM/SLAM.

**References verified (Crossref, 2026-11-14).** Parkinson & Axelrad 1988, Sturza 1988, Brown 1992, Brenner 1996, Pervan et al. 1998, Joerger et al. 2014 (RAIM); Blanch et al. 2015 (baseline ARAIM, TAES 51(1)), Blanch et al. 2017 (protection levels after exclusion, Navigation 64(4)); Joerger et al. 2018 (GNSS/LiDAR unwanted-object detection, Sensors 18(8)). Characterisations of "solution separation" for Brenner 1996 and of protection level are standard but from memory.

---

## 2. Fix for Eq. 14 / Algorithm 1 (the author's own Sensors 23(23):9309; a self-correction)

Source read via Europe PMC XML (MDPI blocked); typeset math was flattened, so a ≤/≥ could be a dump artefact — **re-check against the PDF**. Prose independently supports the findings.

**Confirmed problems**
1. Eq. 14 and prose: seed pair = **max** entropy. Alg. 1 line 5: `h_ij ≤ h*` (a **min** test).
2. Line 2 sets `h* ← −∞`, so line 5 never fires; C stays empty.
3. Line 11 `h_i ≥ h*` accepts a sensor when entropy rises — opposite of "sharpen the joint PDF" (their Fig. 8: lower entropy = higher precision). Single pass, order-dependent.
4. Eq. 14 is a conditional surprisal, not the differential entropy of a normalised PDF.
5. Maximum number of faulty sensors never stated; independence assumed (Eq. 12) though sensors share one footstep.

**Deeper issue:** entropy of a Gaussian product ignores the mean offset. In a toy (3 good N(0,1) + one sharp biased N(8,0.3²)), fusing the faulty sensor lowers entropy most (0.87 → 0.095). Entropy alone cannot detect a Byzantine sensor.

**Proposed formulation**
- F_C = ∏_{i∈C} f_i; Z_C = ∫ F_C; p_C = F_C/Z_C; H(C) = −∫ p_C log p_C. Assume #faulty ≤ F < m/2. Work in log space.
- Admission: Λ_k(C) = |S| · Z_{C∪k}/Z_C ≥ 1 (Bayes factor vs uniform null). Alternative: Bhattacharyya coefficient ≥ τ.
- Seed: pair maximising Z_ij (not entropy). Expand: add feasible k with largest Λ_k (order-independent). Stop when none admissible.
- Select: among admissible subsets with ≤ F removals, minimise H(C).
- Entropy then measures precision of the surviving consensus; Λ provides the fault test.

**Equivalence warning.** For the unnormalised-product surprisal, −log Z = ½νᵀS⁻¹ν + ½log det(2πS): a leave-one-out Mahalanobis test plus a volume term. Solution separation equals the leave-one-out innovation statistic when H is invertible. So under Gaussian residuals the entropy criterion ≈ leave-one-out χ². The paper must show where it adds something (heteroscedastic noise, non-Gaussian/multimodal residuals, occlusion, grazing) or concede equivalence (pivot P1 in section 7).

---

---

---

## 3a. Measurement model (the gate is derived from it)

Nominal measurement per view: **d̂ = a·d + b + ε**. Explicit uncertainty sources:
- **U1** scale/offset ambiguity (a, b): nuisance; prior around (1,0) for a metric model, none for affine-invariant; absorbed by alignment.
- **U2** heteroscedastic noise σ(range, texture, grazing angle) (e.g. log σ = γ₀ + γ₁ log r + γ₂ g(τ) + γ₃ tan²θ).
- **U3** spatial correlation: Σ_ε = D C D, C = (1−ν)K_ℓ + νI, Gaussian kernel with length scale ℓ (pixels) and nugget ν.
- **U4** pose error ξ ~ N(0, Σ_q) (from the provisional pose graph's marginal covariances) → J_q Σ_q J_qᵀ (rank ≤ 6).
- **U5** small intrinsics error ζ ~ N(0, Σ_K) → J_K Σ_K J_Kᵀ (larger if the model estimates the focal length, e.g. Depth Pro).
- **U6** surface uncertainty Σ_surf = a²·diag σ²(u): the composite variance of the closed-form surface (diagonal by assumption; ignores spatially correlated surface error and shared per-view pose error; jackknife only in the MCMC ablation).
- Not modelled: occlusion (mask), dynamics, non-Gaussian tails (absorbed by threshold calibration).

Residual against the rendered surface: r₀ ≈ Aη + ε + J_qξ + J_Kζ + e_S, **Σ_r = Σ_ε + J_qΣ_qJ_qᵀ + J_KΣ_KJ_Kᵀ + Σ_surf**, A = [d, 1]. Whitening/projection: M = Σ_r⁻¹ − Σ_r⁻¹A(AᵀΣ_r⁻¹A)⁻¹AᵀΣ_r⁻¹. Σ_r = scaled stationary kernel + low-rank (≤ 6+p, plus a diagonal) ⇒ Woodbury + FFT/CG, near-linear in n; subsample to the correlation length if needed. **A fault is any deviation φ with Mφ ≠ 0** (what the nominal model cannot explain).

**Calibration.** Fit σ(·), ℓ, ν **once per depth model on clean dev data** (max-likelihood of residuals vs laser reference depth, no surface term); Σ_q from the provisional pose graph; Σ_surf online from the composite variance. Or use a model that outputs confidence as a covariate of σ(·). **Public READMEs, checked 2026-11-14:** Metric3D v2 (ViT) returns depth and normal confidence; UniDepthV2 lists a confidence output; **Depth Pro (default) returns depth + focal length only → fit σ(·)**; MoGe returns a binary validity mask, not a confidence; Depth Anything V2 and Marigold document none (Marigold has a multi-pass ensemble option). Adequacy check (H6): on fault-free views mean T/(n−2) = 1±0.1 and false alarm within [0.5α, 1.5α] without per-scene retuning.

## 3. Gate: statistics, detection theory, surface consensus, alarms, protection level

- Aligned depth of view i: `D~_i = s_i·D_i + t_i` (prior (1,0) for a metric model); healthy `D~ = d + ε`, `σ(d) = σ0 + κ·d` fitted on dev data.
- Residual weight with pose-induced pointing error: `v_res = σ_r² + (r σ_φ tanθ)²` (regularised, gate θ>85°); normal-direction variance `v_n` floors the primitive extent and bounds σ_h². Derivations in manuscript Appendix A.
- **Shared surface (v2.5):** depth maps only (colour out of scope); Gaussians over an SDF for closed-form construction and built-in uncertainty. Primitive = extent (μ, n, tangent half-widths s_u,s_v; rendering only) + uncertainty (offset h=nᵀμ, σ_h²=nᵀPn, λ=1/σ_h², P = J_dσ_d²J_dᵀ+J_qΣ_qJ_qᵀ+J_KΣ_KJ_Kᵀ) + opacity α; extent separate from uncertainty so confident primitives do not shrink out of the render. Forward: r=K⁻¹[u v 1]ᵀ, μ=R d r+t, n from the depth gradient, s=d/f stretched by 1/cosθ. **Fusion in information form:** λ=Σλ_i, h=Σλ_ih_i/λ, extent = union (never shrinks); **exact leave-view-j-out by subtracting λ_j (= RAIM solution separation = leave-one-out innovation, proved/tested)**; no folds, no refits. Backward to view j: d_k=(h_{-j}−nᵀc_j)/(nᵀR_j r_u), σ_k²=σ_h²/cos²θ_j + pose term, composite with extent-only weights w_k=α_kG(u;Σ_ext)T_k, d(u)=Σw_kd_k/Σw, σ²(u)=Σw_kσ_k²/Σw; residual r=d̂−(a d(u)+b), Var=σ_d²+a²σ²(u). **Rendering informs fusion:** projective association (render into the new view; ray hits primitive k within the gate → update k, else spawn), visibility from transmittance, soft weight w_kλ_i, render-and-compare gate decides entry. No adaptive densification; fixed-budget 3DGS-MCMC is an ablation. Each region outputs posterior, protection level and view count (availability; <3 views = untestable). Code: `src/sensorfault/eval/surface_model.py`, tests `tests/core/test_surface_model.py`.
- Held-out depth map: robust affine fit to the rendering; pixels with support α≥τ_α, valid, below the grazing gate.
- **Statistics (on the whitened, nuisance-projected residual):** energy T = r₀ᵀMr₀ ~ χ²(n−2) (any pattern, incl. zero-mean warps); matched region scan z_j = g_jᵀMr₀/√(g_jᵀMg_j) over R candidate patterns with Bonferroni threshold; evidence (mean log Bayes factor, corrected admission test; ranks like χ² on whitened residuals). Thresholds by clean-run permutation to α=5% per view. n_eff = tr(C_r)²/tr(C_r²) is *computed* from Σ_r.
- **Outputs:** g_i; no-consensus alarm (surface does not support the view / too few other views); scene-level alarm (>½ rejected); availability reported next to false-alarm rate.
- **Per-primitive protection level (closed form):** PL_h = max_j (λ_j/λ)(z_α+z_β)√(1/λ_j+1/λ_{−j}) + z/√λ; equal information over m views: (z_α+z_β)/√(λ(m−1)) + z/√λ; untestable with <3 views.
- **Conditional protection level for general scalar outputs (H9, optional, first cut):** PL_q = max_i √(λ₀·s_iᵀΣ̃_r s_i) + z_q·σ_q, Σ̃_r = Σ_r − A(AᵀΣ_r⁻¹A)⁻¹Aᵀ, λ₀ = noncentrality of the energy test at missed-detection β (ellipsoid of undetected faults φᵀMφ ≤ λ₀; tests verify the maximiser). iid case: √λ₀·σ·‖s_⊥‖ = slope × MDB. s_i from perturb-and-refit. Assumptions A1 single faulty view, A2 Σ_r calibrated, A3 linear response, A4 uncorrupted surface. **Conditional on a fault being present** (no prior fault probabilities for learned depth); coverage is *tested*, expected to fail when A1/A4 fail. Energy-test-only bound is conservative; the scan tightens it.
- **Cost:** pairwise `C_pair ≈ N·d̄·P·c_w`; closed-form surface `C_surf ≈ N·P·(c_b+c_r)` (build + render, no iterations, O(1) leave-view-out per primitive); break-even `d̄* ≈ (c_b+c_r)/c_w`; pairwise cheaper on sparse graphs; primitive count grows with the scene. Pairwise gate = warp neighbours into view i, build their consensus with the corrected set-growing criterion, test view i against it (manuscript Appendix C).
- Caveats: provisional poses corrupt surface and gate; faulty views dominating a region make the surface follow the fault; model-level faults are undetectable by construction; information fusion assumes independent observations (overconfident under within-view correlation and shared per-view pose error); σ_h² ignores normal uncertainty; Σ_surf is diagonal; association and cold start are order dependent; a faulty view with a large offset can spawn separate, untestable primitives; planar primitives fit thin or curved structure poorly.

---

## 4. Baselines and reference points (configs/baselines/)

| Family | Methods | Registry name (stub) |
|---|---|---|
| Robust kernels (no gate) | Huber, Cauchy, GNC (Geman–McClure); switchable constraints, DCS, max-mixture (loop closures) | `robust_kernel_pgo` |
| Soft view weights | per-view Student-t / learned-scale weight in the differentiable PGO (no exclusion, no alarm) | `soft_view_weights_pgo` |
| Pixel filters | COLMAP geometric consistency (`consistency_voting` numpy test exists); same flags aggregated into a count test | `colmap_consistency_filter`, `pixel_count_test` |
| Voting | per-pixel median/majority across the views that observe the pixel (`median_fusion` exists); **cross-model** only in H10 | `plain_voting` (param `rule`) |
| Floor / bound / context | no gate; oracle gate; estimate-and-correct | `no_gate`, `oracle_gate`, `estimate_and_correct` |
| Gate variants | surface vs pairwise × {energy (whitened, projected), region scan, combined, evidence}; ablations: no leave-view-out (joint surface), single covariance (extent = uncertainty), fixed-budget 3DGS-MCMC surface with explicit refits | `view_gate` (params `consensus`, `statistic`, `surface`, `leave_view_out`, `densification`, `sources`) |
| Reference points | MASt3R-SfM, VGGT, COLMAP SfM | `mast3r_sfm`, `vggt`, `colmap_sfm` |

Methods that do not exclude (kernels, soft weights, pixel filters) are compared on detection by an **action rate** calibrated to the same 5% false-alarm rate. Same images, matches, depth maps, fault realisations and solver for all; thresholds tuned on dev scenes. **All are stubs** (`src/sensorfault/baselines/gate_testbed.py`); the test bed is not implemented.

---

## 5. Hypotheses (pre-register at week 1; mirrored in configs/h*.yaml and the manuscript)

- **H1 (regime map; key figure).** Sweep per-pixel bias δ∈{0.1..4}σ × fraction **f∈{0.01..0.8}**, plus the **affine control (f=1; bias ∝ σ(d))**. (a) empirical MDB within ×2 of β_min computed from the *measured* Σ_r; (b) sub-noise regime (δ≤0.5, 0.2≤f≤0.8): gate ≥80% detection, best pixel filter / kernel / soft weight each ≤20%; (c) sparse-outlier regime (δ≥4, f≤0.02): best pixel filter ≥ gate (gate not expected to win); (d) **affine control: not detected (power ≤ α+5 points) and not harmful (error within 5% of clean)**. *Falsified if* MDB off by >×2; gap at δ=0.3,f=0.5 <30 points; the gate beats the pixel filter by ≥10 points in (c); or the uniform bias is detected or degrades the output.
- **H2 (consequence).** Sub-noise regime: gating cuts ATE and fused-surface error ≥25% vs the best of {kernels, soft weights, COLMAP filter, voting}, within 20% of oracle; sparse-outlier regime: baselines tie or win.
- **H3 (statistic).** (a) evidence ≥ energy by ≥10 points only where chi-square assumptions break (else tie → pivot P1); (b) region scan beats energy by ≥20 points for a region-confined bias, energy beats scan by ≥20 points for a diffuse smooth warp, combined within 3 points of the better.
- **H4 (leave-view-out, alignment, support).** Surface with the view subtracted + alignment + support mask: clean false alarm ≤5%; joint surface lowers power ≥10 points; no support mask doubles false alarm.
- **H5 (placement, soft down-weighting, complementarity).** Gate before PGO beats the same statistic after PGO and beats soft per-view down-weighting in the sub-noise regime; gate + kernel ≥ either. *Falsified if soft down-weighting suffices.*
- **H6 (adequacy of the measurement model, price of monitoring, reference points).** Adequacy: mean T/(n−2) = 1±0.1 and false alarm in [0.5α,1.5α] with σ(·), ℓ, ν fitted once on dev scenes (else the model is misspecified and H1a cannot be trusted). Clean data: false alarm ≤5%, availability ≥90%, error change ≤5%; natural failures: gap to MASt3R-SfM/VGGT narrows (context only).
- **H7 (shared surface vs pairwise).** (a) linear scaling vs N·d̄, ≥2× faster at N≥100, d̄≥10 (pairwise cheaper when sparse; report break-even); (b) power within 5 points of pairwise; (c) closed form within 3 points of the fixed-budget MCMC surface with explicit refits, at a fraction of the cost; per-region posterior, PL and view count with reported coverage; (d) robust to view order (cold start / association failure measured).
- **H8 (pose faults).** (a) ≥80% detection above the implied pose MDB, exclusion lowers ATE vs kernels; (b) **large false loop closures: loop-closure baselines match the gate within 5 points (gate not expected to win)**; (c) small coherent offsets (≤0.5°): gate beats kernels by ≥20 points.
- **H9 (protection level; optional).** Single-fault coverage of the ellipsoid PL computed from the measured Σ_r: exceedance ≤β+5 points under A1–A4; >2β when assumptions fail.
- **H10 (ensemble ablation).** (a) models fail together (lift ≥2, phi ≥0.3); (b) where ≥½ of the models fail cross-model voting detects ≤20% while the single-model gate detects ≥20 points more (converse reported); (c) ensemble gate gains ≥5 points or the M× cost is not justified.
- **Breakdown (honest negatives).** Model-level fault (undetectable); very few views (no consensus); faults below MDB; small n_eff (smooth/textureless scenes); **misspecified measurement model (ℓ off ×2, σ(·) off ×1.5, no nugget, heavy tails, tiny K)**; sparse outliers (baselines win/tie); low overlap; faulty views dominating a region or two faulty views in a region; **large false loop closures**; poor provisional poses.

Mapping from v2.2: old H1 → H2 (consequence) + new H1 (regime map); old H2 → H3(a) (+ new region-scan/energy part); old H3 → H4; old H4 → H5 (+ soft down-weighting); old H5 → H6 (+ availability); old H6 → H7; new H8 (pose faults), H9 (protection level); old H7 → H10.

---

## 6. Data and fault protocol (configs/datasets/, configs/faults/view_fault_grid.yaml)

| Dataset | Role | Notes |
|---|---|---|
| ETH3D | primary real | multi-view images, laser-scan GT depth, calibrated poses; natural failures (glass/reflective, sky, textureless) labelled against laser depth. **[UNVERIFIED]** scene names in base.yaml, licence, download script not written. |
| ScanNet++ | secondary | laser-scanned geometry + registered images; reflective/transparent surfaces; access terms. |
| Replica | controlled synthetic | exact depth; vary overlap and fault parameters; research-only; release generator, not renderings. |
| TUM RGB-D, Tanks & Temples | metric definitions | ATE/RPE and reconstruction metrics. |
| H7 scaling study | N∈{25,50,100,200}, d̄∈{2,5,10,20} | Replica renderings, subsampled ScanNet++ sequences, ETH3D multi-camera videos. **[UNVERIFIED]** views per scene and achievable d̄. |

Injected faults (deterministic, seeds `SeedSequence([master, hash(view id), fault id])`, ≥5 masters): coherent bias δ×f with f ≤ 0.8 (contiguous or scattered mask, both signs) plus the f=1 / ∝σ affine control; smooth warp; region failure; non-affine distortion; spoofed pose prior (1–10 cm, 0.1–1°, constant or drifting, block of views); intermittent glitches (10–50 cm, 2–10°); false loop closures (1,3,5 spurious edges); p∈{10,20,40,60}% of views scattered/clustered; controls: sparse outliers, clean. Ensemble ablation: 1..⌈M/2⌉ models corrupted on the same views, independent vs identical. **Injectors for v2 are not implemented** (v1 injectors cover depth bias/extrinsic drift on physical sensors).

Metrics: integrity first — detection power at 5% false alarm (heatmap over δ×f), empirical MDB vs theory, false-alarm and missed-detection rates, no-consensus rate, availability, elimination-curve AUC, PL exceedance; then ATE/RPE/pose AUC, fused-surface accuracy/completeness/F-score, depth AbsRel/δ1; per-stage runtime/memory/log–log slope. Scene-level bootstrap; paired Wilcoxon.

**Compute notes.** One model inference per view (N per scene, cached), one closed-form surface build and render per view (no optimisation), one pose graph; ensemble ablation costs M× inference (subset of scenes); H9 sensitivities need perturb-and-refit (subset). Risks **[UNVERIFIED]**: Depth Pro (and alternatives) memory on 8 GB at the chosen resolution; primitive count growth with scene size; MASt3R-SfM/VGGT memory (subsets). Mitigation: cache depth, lower surface resolution, subsample pixels to the correlation length.

---

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | Re-read the Sensors PDF equations; verify datasets/licences; Depth Pro licence and 8 GB memory; dev-scene comparison Depth Pro / UniDepth / Metric3D v2 vs laser; prereg tag | M1 |
| 2 | Depth runner (one model per view); cache depth on ETH3D dev scenes; per-view error vs laser (natural failure labels); **go/no-go: measure σ(range, texture, grazing), the correlation length ℓ and the nugget ν on clean residuals (fit once, per the calibration protocol); assemble Σ_r (pose from the provisional graph, Σ_surf from the closed-form composite variance); compute n_eff = tr(C)²/tr(C²) and κ for the regime-map patterns; check adequacy (mean T/(n−2) = 1±0.1, false alarm within [0.5α,1.5α]). Go if the predicted MDB at (δ=0.5, f=0.5) is ≤0.5σ and the model is adequate; no-go/pivot P2 if n_eff·f(1−f) is so small that the sub-noise regime is empty, or if the model cannot be made adequate.** | M2 |
| 3 | Matches; geometry-only graph (provisional poses); metrics harness | M3 |
| 4 | Differentiable PGO; baselines: kernels, soft weights, COLMAP filter, pixel-count test, voting | M4 |
| 5 | View-level fault injectors (bias grid f≤0.8 + affine control, pose faults); closed-form Gaussian surface builder (information fusion, exact leave-view-out, ray-plane render); whitened energy + region-scan gate | M5 |
| **6** | **Gate 1:** H1 regime map on dev scenes (sub-noise corner and sparse corner) | go / pivot |
| 7 | Evidence gate; H3; H4 | |
| 8 | H2, H5 (placement, soft down-weighting); H7 scaling vs N and d̄ | |
| 9 | Pose faults (H8); breakdown scenarios; **Gate 2** | honest negatives |
| 10 | Natural failures; reference points; H6 | |
| 11 | H9 protection level (optional); ScanNet++/Replica sweeps (cut if behind) | |
| 12 | H10 ensemble ablation; gaps; runtime table | |
| 13 | **Experiment freeze** | tag |
| 14–16 | Draft, figures, appendix, reproducibility package, AI-use disclosure | submission-ready |
| 17–18 | Buffer | |

**Gate 1:** go if the gate beats the baselines in the sub-noise corner of H1 and the measured MDB is within ×2 of theory. Pivot P1: evidence ≈ chi-square → chi-square statistic, entropy relation as equivalence result. Pivot P2: no sub-noise advantage (H1b fails) or soft down-weighting suffices (H5) → analysis paper ("when do pixel filters, kernels and soft weights suffice?"). Pivot P3: surface not cheaper or less reliable than pairwise (H7) → keep pairwise, report surface as negative result. **Early risk:** real residuals far from the Gaussian model (heavy tails, strong correlation, mis-calibrated σ(·)) raise the MDB, inflate the false-alarm rate and shrink the gate's regime; this is checked in week 2, before any fault experiment.
**Cut order:** H9 protection level → ScanNet++/Replica sweeps → scene-level alarm refinements → exact leave-one-view-out reference → estimate-and-correct → H10 scenes. **Never cut:** the H1 regime map, breakdown scenarios, clean-data cost (H6), the soft down-weighting baseline (H5), the Eq. 14/Alg. 1 fix.

---

## 8. Paper outline (follows the author's own papers: Sensors 23(23):9309 and 24(23):7744)

| Section | Follows | Says |
|---|---|---|
| I Introduction (RAIM analogy; core argument; A Relevant Literature incl. integrity monitoring/FDE; B Challenges C1–C7; C Baseline Study; D Contributions; E Organization) | 2023 and 2024 introductions | learned depth needs integrity monitoring; sub-noise coherent faults are invisible per pixel and visible per view; down-weighting gives no trust statement |
| II Method (A Definitions and Notation; B Forward Problem incl. depth and pose faults; C Inverse Problem; D View-Level Gate: requirements, surface consensus, residuals, **detection statistics and MDB**, corrected admission, cost, exclusion/alarm/availability, **conditional protection level**; E Ensemble Variant; F Test Bed) | 2023 Method (BSE) and 2024 forward/inverse split | the gate as an integrity monitor |
| III Experiments (A Setup: data, depth model, fault injection, baselines, metrics; B Hypotheses H1–H10 and breakdown; C Implementation) | 2023 Setup/Implementation | where gating helps and where it does not |
| IV Results | both | regime map first |
| V Conclusions and Future Work (incl. Limitations) | 2023 | no separate Discussion |
| Appendices A–D | 2023 | variance; proof; pairwise gate; detection analysis |

---

## 9. IEEE Access facts (verified on official pages 2026-11-13)
Verified on official pages: gold OA; APC $2,160 (+ taxes; 5%/20% member discounts only); no page limit, <20 pp recommended; single-anonymised, ≥2 reviews, ~4–6 weeks submission-to-publication, acceptance ~20%; reviewers judge technical soundness, not necessarily high novelty; no formal data/code statement required; preprints allowed (arXiv named in IEEE-wide policy, not on Access pages); AI-assisted writing must be disclosed in Acknowledgments; fabricated references → rejection; public ORCID required.
Not verified / yours to resolve: Virginia Tech–IEEE read-and-publish agreement (ask VT libraries; VT OASF announced closed Jul 2025); current IEEE copyright-form (eCF) wording and any publication-approval obligations that apply to you — check the live form. No Access conflict-of-interest page found. Not legal advice. Comparable-paper titles came from catalogue metadata only.

---

---

---

## 10. Open items
- [ ] Re-read the Sensors 23(23):9309 PDF: Alg. 1 inequality directions, entropy normalisation (author to confirm).
- [ ] Author to confirm the reading of "partly end-to-end" / "differentiable, renderable PGO".
- [ ] **Week 2 go/no-go: fit σ(range, texture, grazing), ℓ, ν on clean residuals; assemble Σ_r; compute n_eff and κ; check adequacy** — the theory's weak point is the measurement model.
- [ ] Verify ETH3D scene names/licence, ScanNet++ terms, the default model's weights licence and GPU memory; views per scene and achievable d̄ for H7.
- [ ] Literature pass: integrity monitoring for visual/learned navigation; depth-prior filtering in SfM/SLAM; uncertainty-aware monocular depth in pose graphs.
- [ ] Implement: fault injectors (bias grid, pose faults); pixel-wise gate; soft-weight baseline; test bed pipeline; ETH3D download script.
- [ ] Choose and cite the matcher and solver; fix association gate (τ_h, τ_ext, cos_min), footprint, τ_α, s_min on dev scenes (N_s, n_it for the MCMC ablation only).
- [ ] Affiliation / funding / AI-use disclosure in the manuscript (author's decision).

## 11. Change log
- 2026-11-13: v2 design (gate = contribution, pipeline = test bed).
- 2026-11-14: v2.1 shared splat surface restored; v2.2 one metric depth model per image, ensemble → ablation.
- 2026-11-14: **v2.3 integrity-monitoring framing (RAIM-style FDE).** Core argument made explicit (sub-noise per pixel, coherent per view; MDB ∝ 1/(f√n_eff)); pose faults added; soft down-weighting is a baseline; conditional protection level added (optional); hypotheses reorganised to H1–H10 (new H1 regime map, H8 pose faults, H9 protection level; ties/losses pre-registered where gating should not help); `detection_theory` module + tests; configs and tests regenerated; five RAIM/integrity references added.
- 2026-11-14: **v2.4 explicit measurement model.** Nominal d̂ = a·d + b + ε with scale/offset ambiguity, heteroscedastic spatially-correlated noise (σ(range, texture, grazing), ℓ, nugget ν), pose, intrinsics and surface uncertainty (jackknife over K cross-fitted surfaces) → Σ_r; tests on whitened, nuisance-projected residuals; n_eff = tr(C)²/tr(C²) computed; MDB, scan and protection level from Σ_r. **Correction:** a bias uniform over a whole view (f=1) or ∝σ(d) is an offset, absorbed by the alignment — not a fault, not detected, not harmful; the regime map now uses f ≤ 0.8 and an f=1 affine control (H1d); the signed-mean statistic is dropped in favour of a region scan; the earlier MDB table is withdrawn. New: calibration protocol and public-README check of which models output confidence; adequacy test in H6; misspecification breakdown; week-2 go/no-go is now measuring σ(·), ℓ, ν. Code: tested measurement-model theory (nugget correlation grid, Σ_r assembly, jackknife surface covariance, projected precision, κ, MDBs, n_eff, pose Jacobian, ellipsoid PL).
- 2026-11-14: **v2.5 surface reframe.** Depth maps only (colour dropped; the photometric-term ablation and future work removed); Gaussian planar primitives over an SDF; extent separated from uncertainty; closed-form forward (σ_h²=nᵀPn from depth, pose and intrinsics); information-form fusion with exact leave-view-out by subtraction (= solution separation = innovation); ray-plane backward render with alpha compositing and extent-only weights; Σ_surf = a²diag σ²(u) replaces the K-fold jackknife; projective association with soft weights; per-region posterior, protection level and view count; adaptive densification dropped, fixed-budget MCMC kept as an ablation only. H4 reworded (leave-view-out vs joint), H7 revised (closed form vs pairwise vs MCMC fixed budget; association/view order) — the request's "H6" for the shared-surface hypothesis is H7 in this numbering. New breakdown `association_failure`; configs `gate_surface_mcmc_fixed_budget`, `gate_surface_single_covariance`; code `surface_model.py` (12 tests) and `sigma_surf_diag`. Schedule: week 5 builder is closed form.
