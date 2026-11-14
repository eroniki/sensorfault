# View-level fault detection and exclusion for learned-depth reconstruction — research plan (design v2.3)

Target: IEEE Access, solo author, 14–18 evening weeks (~10 h/wk), one 8 GB consumer GPU (RTX 2070 on this machine).
Title (fixed): *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*.
Status: v2.3 of 2026-11-14. Framing = **integrity monitoring (RAIM-style FDE) on views**. Kept from v2.2: one metric depth model per image (default Depth Pro), shared splat-surface consensus (cross-fitted, fixed budget, doubles as fused output), "gate is the contribution, pipeline is the test bed". Design v1 (physical-sensor faults) is in `configs/legacy_sensor_faults/`. Items marked **[UNVERIFIED]** come from memory or abstract-only reads.
Hygiene: public datasets, public pretrained models and public code only.

---

## 1. Framing, core argument and honest gap

**RAIM ↔ this paper.**

| RAIM / FDE | Here |
|---|---|
| redundant satellite ranges | other views of the same scene (a single model gives no redundancy within a view) |
| fault: a faulty satellite | view-level fault: depth bias/warp/failed region **or** pose fault (spoofed prior, glitch, false loop closure) |
| consistency test on residuals (range comparison, parity, residual, solution separation) | per-view test of depth vs the rendered shared surface; mean-shift + energy statistics |
| threshold from false-alarm probability | per-view false-alarm α (5%), threshold calibrated on clean dev data |
| minimum detectable bias for missed-detection β | δ_min = (z_α+z_β)/(f·√n_eff) |
| exclusion | g_i = 0 (drop the view's depth from PGO and fusion) |
| alert / loss of integrity | no-consensus alarm (unsupported view) and scene-level alarm (>½ views rejected) |
| availability | fraction of views tested and kept |
| protection level | conditional PL = max_i |s_i|·δ_min,i·σ + z_q·σ_q (single fault; assumptions A1–A4) |

**Core argument.** A coherent view-level fault adds a small bias to *every pixel* of one view. At sub-noise per-pixel size (e.g. 0.3σ) a pixel filter (|ρ|>k=3) sees almost nothing: flagging probability 0.0040 vs 0.0027 null. Aggregated over the view's n_eff effectively independent co-visible pixels the same bias shifts the mean by fδ√n_eff standard errors. **The gate wins exactly when faults are sub-noise per pixel but coherent across a view** (glass, sky, out-of-distribution scenes). It should *not* win when a fault is large but confined to few pixels (pixel filters are the right tool) or for large false loop closures (switchable constraints / DCS / max-mixtures were built for them).

**Verified numbers (code: `src/sensorfault/eval/detection_theory.py`, tests in `tests/core/test_detection_theory.py`; Monte Carlo cross-checked).**
- δ=0.3, k=3: pixel flag prob 0.00395 vs 0.0027 null; second-order excess kφ(k)δ².
- n_eff=1000: mean-shift SNR 9.49·f vs aggregated pixel-count SNR 0.76·f — ratio ≈12.4. The count overtakes the mean shift only above δ≈1.65σ (k=3).
- MDB (α=0.05, β=0.1, z_α+z_β=3.24), n_eff=1000: 0.10σ (f=1), 0.51σ (f=0.2), 2.05σ (f=0.05). MDB ∝ 1/(f√n_eff). Energy statistic (zero-mean warps): MDB ∝ n_eff^{-1/4} (0.38σ at n_eff=1000, f=1); mean-shift is blind to zero-mean warps (MC power 0.118 vs energy 0.998 at rms 0.5σ).
- **The weak point is n_eff**: learned-depth residuals are spatially correlated, heavy-tailed and heteroscedastic; n_eff and the threshold must be calibrated on clean data. The Gaussian analysis is a guide, not a guarantee.

**Why not robust kernels / soft down-weighting.** They down-weight when a residual is large (a sub-noise coherent bias passes through) and give no trust statement: no exclusion, no false-alarm/missed-detection probability, no alarm. Soft per-view weights inside the differentiable PGO are the closest relative and are a **baseline (H5)**; if they match the gate in the sub-noise regime the paper's premise fails — H5 is written to find out.

**Fault models.** Depth: coherent bias (δ, f — the regime map), smooth warp, region failure, non-affine distortion. Pose: spoofed pose prior (consistent/drifting offset on a block), intermittent glitches, false loop closures. A pose error shifts a whole view's residual through the same pointing-error term (r·δφ·tanθ), so one test covers both; *isolating* depth-vs-pose is future work.

**Test bed (not the framing).** RGB → one metric depth model per image (default **Depth Pro**: absolute-scale depth, no intrinsics needed, fast 2.25 MP in 0.3 s per its authors, sharp; alternatives UniDepth, Metric3D v2; chosen on dev scenes only) → matches → provisional poses → shared splat surface fitted from other views (K folds, fixed N_s, 3DGS-MCMC relocation) → gate → **differentiable, renderable PGO** (optional render-based depth constraint; soft down-weighting = baseline) → surface refitted on kept depth = fused output. Ensemble of models = ablation only (H10).

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

## 3. Gate: statistics, detection theory, surface consensus, alarms, protection level

- Aligned depth of view i: `D~_i = s_i·D_i + t_i` (prior (1,0) for a metric model); healthy `D~ = d + ε`, `σ(d) = σ0 + κ·d` fitted on dev data.
- Residual weight with pose-induced pointing error: `v_res = σ_r² + (r σ_φ tanθ)²` (regularised, gate θ>85°); normal-direction variance `v_n` initialises splat covariances. Derivations in manuscript Appendix A.
- **Shared surface:** N_s Gaussians, fixed budget (3DGS-MCMC relocation), depth-only fit (robust kernel + accumulated-alpha + anisotropy cap + depth-distortion/normal-consistency). **Cross-fitting:** K folds of neighbouring views; S_{-f} fitted outside fold f, rendered in f. A view never meets a surface fitted to it.
- Held-out depth map: robust affine fit to the rendering; pixels with support α≥τ_α, valid, below the grazing gate.
- **Statistics:** mean-shift z = √n_eff·ρ̄ (uniform bias), energy T (zero-mean warps), evidence (mean log Bayes factor, corrected admission test). n_eff from residual autocorrelation on clean data; thresholds by clean-run permutation to α=5% per view.
- **Outputs:** g_i; no-consensus alarm (surface does not support the view / too few other views); scene-level alarm (>½ rejected); availability reported next to false-alarm rate.
- **Conditional protection level (H9, optional, first cut):** PL_q = max_i |s_i|·δ_min,i·σ + z_q·σ_q with s_i from perturb-and-refit. Assumptions A1 single faulty view, A2 calibrated σ and n_eff, A3 linear response, A4 uncorrupted surface. **Conditional on a fault being present** (no prior fault probabilities for learned depth, so no integrity-risk allocation); coverage is *tested*, and expected to fail when A1/A4 fail.
- **Cost:** pairwise `C_pair ≈ N·d̄·P·c_w`; surface `C_surf ≈ ((K+1)·n_it+1)·N·P·c_r`; break-even `d̄* ≈ ((K+1)·n_it+1)·c_r/c_w` — linear in N but a large constant; pairwise cheaper on sparse graphs. Pairwise gate = warp neighbours into view i, build their consensus with the corrected set-growing criterion, test view i against it (manuscript Appendix C).
- Caveats: provisional poses corrupt surface and gate; faulty views dominating a region make the surface follow the fault; model-level faults are undetectable by construction.

---

## 4. Baselines and reference points (configs/baselines/)

| Family | Methods | Registry name (stub) |
|---|---|---|
| Robust kernels (no gate) | Huber, Cauchy, GNC (Geman–McClure); switchable constraints, DCS, max-mixture (loop closures) | `robust_kernel_pgo` |
| Soft view weights | per-view Student-t / learned-scale weight in the differentiable PGO (no exclusion, no alarm) | `soft_view_weights_pgo` |
| Pixel filters | COLMAP geometric consistency (`consistency_voting` numpy test exists); same flags aggregated into a count test | `colmap_consistency_filter`, `pixel_count_test` |
| Voting | per-pixel median/majority across the views that observe the pixel (`median_fusion` exists); **cross-model** only in H10 | `plain_voting` (param `rule`) |
| Floor / bound / context | no gate; oracle gate; estimate-and-correct | `no_gate`, `oracle_gate`, `estimate_and_correct` |
| Gate variants | surface vs pairwise × {energy, mean-shift, combined, evidence}; ablations: no cross-fit, adaptive densification, exact leave-one-view-out | `view_gate` (params `consensus`, `statistic`, `cross_fit`, `densification`, `sources`) |
| Reference points | MASt3R-SfM, VGGT, COLMAP SfM | `mast3r_sfm`, `vggt`, `colmap_sfm` |

Methods that do not exclude (kernels, soft weights, pixel filters) are compared on detection by an **action rate** calibrated to the same 5% false-alarm rate. Same images, matches, depth maps, fault realisations and solver for all; thresholds tuned on dev scenes. **All are stubs** (`src/sensorfault/baselines/gate_testbed.py`); the test bed is not implemented.

---

## 5. Hypotheses (pre-register at week 1; mirrored in configs/h*.yaml and the manuscript)

- **H1 (regime map; key figure).** Sweep per-pixel bias δ∈{0.1..4}σ × fraction f∈{0.01..1}. (a) empirical MDB within ×2 of δ_min; (b) sub-noise regime (δ≤0.5, f≥0.2): gate ≥80% detection, best pixel filter / kernel / soft weight each ≤20%; (c) sparse-outlier regime (δ≥4, f≤0.02): best pixel filter ≥ gate (gate not expected to win). *Falsified if* MDB off by >×2; gap at δ=0.3,f=1 <30 points; or the gate beats the pixel filter by ≥10 points in (c).
- **H2 (consequence).** Sub-noise regime: gating cuts ATE and fused-surface error ≥25% vs the best of {kernels, soft weights, COLMAP filter, voting}, within 20% of oracle; sparse-outlier regime: baselines tie or win.
- **H3 (statistic).** (a) evidence ≥ energy by ≥10 points only where chi-square assumptions break (else tie → pivot P1); (b) mean-shift beats energy by ≥20 points for uniform bias, energy beats mean-shift by ≥20 points for zero-mean warps, combined within 3 points of the better.
- **H4 (cross-fitting, alignment, support).** Surface fitted without the view + alignment + support mask: clean false alarm ≤5%; joint fit lowers power ≥10 points; no support mask doubles false alarm.
- **H5 (placement, soft down-weighting, complementarity).** Gate before PGO beats the same statistic after PGO and beats soft per-view down-weighting in the sub-noise regime; gate + kernel ≥ either. *Falsified if soft down-weighting suffices.*
- **H6 (price of monitoring, reference points).** Clean data: false alarm ≤5%, availability ≥90%, error change ≤5%; natural failures: gap to MASt3R-SfM/VGGT narrows (context only).
- **H7 (shared surface vs pairwise).** (a) linear scaling vs N·d̄, ≥2× faster at N≥100, d̄≥10 (pairwise cheaper when sparse; report break-even); (b) power within 5 points of pairwise, K∈{3,4,6} within 3 points of exact leave-one-view-out; (c) adaptive densification loses ≥10 points vs fixed budget.
- **H8 (pose faults).** (a) ≥80% detection above the implied pose MDB, exclusion lowers ATE vs kernels; (b) **large false loop closures: loop-closure baselines match the gate within 5 points (gate not expected to win)**; (c) small coherent offsets (≤0.5°): gate beats kernels by ≥20 points.
- **H9 (protection level; optional).** Single-fault coverage: exceedance ≤β+5 points under A1–A4; >2β when assumptions fail.
- **H10 (ensemble ablation).** (a) models fail together (lift ≥2, phi ≥0.3); (b) where ≥½ of the models fail cross-model voting detects ≤20% while the single-model gate detects ≥20 points more (converse reported); (c) ensemble gate gains ≥5 points or the M× cost is not justified.
- **Breakdown (honest negatives).** Model-level fault (undetectable); very few views (no consensus); faults below MDB; small n_eff (smooth/textureless scenes); sparse outliers (baselines win/tie); low overlap; faulty views dominating a region or two faulty views in a region; **large false loop closures**; poor provisional poses.

Mapping from v2.2: old H1 → H2 (consequence) + new H1 (regime map); old H2 → H3(a) (+ new mean-shift/energy part); old H3 → H4; old H4 → H5 (+ soft down-weighting); old H5 → H6 (+ availability); old H6 → H7; new H8 (pose faults), H9 (protection level); old H7 → H10.

---

## 6. Data and fault protocol (configs/datasets/, configs/faults/view_fault_grid.yaml)

| Dataset | Role | Notes |
|---|---|---|
| ETH3D | primary real | multi-view images, laser-scan GT depth, calibrated poses; natural failures (glass/reflective, sky, textureless) labelled against laser depth. **[UNVERIFIED]** scene names in base.yaml, licence, download script not written. |
| ScanNet++ | secondary | laser-scanned geometry + registered images; reflective/transparent surfaces; access terms. |
| Replica | controlled synthetic | exact depth; vary overlap and fault parameters; research-only; release generator, not renderings. |
| TUM RGB-D, Tanks & Temples | metric definitions | ATE/RPE and reconstruction metrics. |
| H7 scaling study | N∈{25,50,100,200}, d̄∈{2,5,10,20} | Replica renderings, subsampled ScanNet++ sequences, ETH3D multi-camera videos. **[UNVERIFIED]** views per scene and achievable d̄. |

Injected faults (deterministic, seeds `SeedSequence([master, hash(view id), fault id])`, ≥5 masters): coherent bias δ×f (contiguous or scattered mask, both signs); smooth warp; region failure; non-affine distortion; spoofed pose prior (1–10 cm, 0.1–1°, constant or drifting, block of views); intermittent glitches (10–50 cm, 2–10°); false loop closures (1,3,5 spurious edges); p∈{10,20,40,60}% of views scattered/clustered; controls: sparse outliers, clean. Ensemble ablation: 1..⌈M/2⌉ models corrupted on the same views, independent vs identical. **Injectors for v2 are not implemented** (v1 injectors cover depth bias/extrinsic drift on physical sensors).

Metrics: integrity first — detection power at 5% false alarm (heatmap over δ×f), empirical MDB vs theory, false-alarm and missed-detection rates, no-consensus rate, availability, elimination-curve AUC, PL exceedance; then ATE/RPE/pose AUC, fused-surface accuracy/completeness/F-score, depth AbsRel/δ1; per-stage runtime/memory/log–log slope. Scene-level bootstrap; paired Wilcoxon.

**Compute notes.** One model inference per view (N per scene, cached), K+1 splat fits per scene, one pose graph; ensemble ablation costs M× inference (subset of scenes); H9 sensitivities need perturb-and-refit (subset). Risks **[UNVERIFIED]**: Depth Pro (and alternatives) memory on 8 GB at the chosen resolution; K+1 splat fits per scene; MASt3R-SfM/VGGT memory (subsets). Mitigation: cache depth, lower surface resolution, small K.

---

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | Re-read the Sensors PDF equations; verify datasets/licences; Depth Pro licence and 8 GB memory; dev-scene comparison Depth Pro / UniDepth / Metric3D v2 vs laser; prereg tag | M1 |
| 2 | Depth runner (one model per view); cache depth on ETH3D dev scenes; per-view error vs laser (natural failure labels); **measure real residual n_eff and tail shape vs the Gaussian theory (go/no-go for H1a)** | M2 |
| 3 | Matches; geometry-only graph (provisional poses); metrics harness | M3 |
| 4 | Differentiable PGO; baselines: kernels, soft weights, COLMAP filter, pixel-count test, voting | M4 |
| 5 | View-level fault injectors (bias grid, pose faults); depth-only fixed-N splat fit with cross-fitting; mean-shift/energy gate | M5 |
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

**Gate 1:** go if the gate beats the baselines in the sub-noise corner of H1 and the measured MDB is within ×2 of theory. Pivot P1: evidence ≈ chi-square → chi-square statistic, entropy relation as equivalence result. Pivot P2: no sub-noise advantage (H1b fails) or soft down-weighting suffices (H5) → analysis paper ("when do pixel filters, kernels and soft weights suffice?"). Pivot P3: surface not cheaper or less reliable than pairwise (H7) → keep pairwise, report surface as negative result. **Early risk:** real n_eff much smaller than n (strong correlation) would raise the MDB and shrink the gate's regime.
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
- [ ] **Measure real residual n_eff, spatial correlation and tails** (week 2) — the theory's weak point.
- [ ] Verify ETH3D scene names/licence, ScanNet++ terms, the default model's weights licence and GPU memory; views per scene and achievable d̄ for H7.
- [ ] Literature pass: integrity monitoring for visual/learned navigation; depth-prior filtering in SfM/SLAM; uncertainty-aware monocular depth in pose graphs.
- [ ] Implement: fault injectors (bias grid, pose faults); pixel-wise gate; soft-weight baseline; test bed pipeline; ETH3D download script.
- [ ] Choose and cite the matcher, solver and splat rasteriser; fix N_s, n_it, τ_α, s_min on dev scenes.
- [ ] Affiliation / funding / AI-use disclosure in the manuscript (author's decision).

## 11. Change log
- 2026-11-13: v2 design (gate = contribution, pipeline = test bed).
- 2026-11-14: v2.1 shared splat surface restored; v2.2 one metric depth model per image, ensemble → ablation.
- 2026-11-14: **v2.3 integrity-monitoring framing (RAIM-style FDE).** Core argument made explicit (sub-noise per pixel, coherent per view; MDB ∝ 1/(f√n_eff)); pose faults added; soft down-weighting is a baseline; conditional protection level added (optional); hypotheses reorganised to H1–H10 (new H1 regime map, H8 pose faults, H9 protection level; ties/losses pre-registered where gating should not help); `detection_theory` module + tests; configs and tests regenerated; five RAIM/integrity references added.
