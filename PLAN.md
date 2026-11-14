# View-level gate for monocular depth sources — research plan (design v2.1, shared splat surface)

Target: IEEE Access, solo author, 14–18 evening weeks (~10 h/wk), one 8 GB consumer GPU (RTX 2070 on this machine).
Status: design v2 of 2026-11-13; v2.1 of 2026-11-14 restores the shared Gaussian-splat surface as the gate's consensus. Design v1 (sensor-level fault elimination on a Gaussian-splat surface, physical depth sensors) is in git history and `configs/legacy_sensor_faults/`. Items marked **[UNVERIFIED]** come from memory or abstract-only reads.
Hygiene: public datasets, public pretrained models and public code only.

**Framing: the gate is the contribution; the pipeline is the test bed.**

---

## 1. Contribution, setting and honest gap

**Contribution.** A *view-level gate* that decides which depth sources to trust *before* pose graph optimisation (PGO) and fusion. A *depth source* is one monocular estimator's depth map for one view (estimator × view). The gate tests each source **once against a single shared Gaussian-splat surface rendered at its view** (instead of against every overlapping view), so its cost grows with the number of views rather than view pairs, and the surface refitted on the admitted sources **doubles as the fused output**. The shared-surface consensus is part of the gate (the contribution), not of the test bed.

**Test bed (answers "where do the depth maps come from").** RGB images → ensemble of M monocular estimators (Depth Anything V2, Metric3D v2, UniDepth, Depth Pro, MoGe; Marigold optional) → feature matches → provisional poses from a geometry-only graph → **shared splat surface, cross-fitted** → **gate** → PGO with gated depth factors (partly end-to-end: learned front-end, differentiable back-end, training-free gate) → surface refitted on admitted sources = fused output (TSDF of the same depth as a secondary check). It is an alternative to classical SfM, not a claimed new system.

**Baselines the gate must beat** (act at other units/stages): robust kernels in the pose graph (per constraint, after the edge exists); COLMAP geometric consistency filter (per pixel, after poses); plain voting (per pixel across sources). **Reference points, not rivals:** MASt3R-SfM, VGGT (and classical COLMAP SfM).

**Why existing methods may not catch a coherent view-level fault (and where they do).**
- Estimators fail coherently (smooth warp, failed region, non-affine distortion), not sparsely; pixels still look plausible.
- Pixel filters decide per pixel; kernels act after the faulty depth has built the constraint; voting needs a majority per pixel.
- Honest limit: a pure scale/shift error is absorbed by the per-source alignment (in the gate and in PGO) and is *not* a fault here. Pixel filters and kernels should tie or win on sparse outliers. Estimators share training data, so common-mode failures can defeat any consensus.
- The gap is regime-specific (clustered or low-neighbour faulty views, smooth faults, fault below per-pixel noise). The paper must show these regimes occur, using natural failures scored against a laser reference.

**Novelty (limited search, to be re-checked).** No work found that gates per depth source before PGO. The earlier novelty table (splat / multi-sensor) is obsolete. New literature areas to check: depth-prior filtering in SfM/SLAM; uncertainty-aware monocular depth in pose graphs; view-graph filtering (1DSfM, GLOMAP are cited). Previous finding kept: UncLe-SLAM (arXiv 2306.11048) is single-stream pixel-wise uncertainty, not multi-source.

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

## 3. Measurement model, shared-surface consensus and gate statistic (as in the manuscript)

- Aligned depth of source (m,i): `D~ = s·D + t`; healthy: `D~ = d + ε`, `σ_m(d) = σ0_m + κ_m·d`, fitted per estimator on dev data.
- Residual weight with pose-induced pointing error: `v_res = σ_r² + (r σ_φ tanθ)²` (regularised: `tan²θ → sin²θ/(cos²θ+ε²)`, ε≈cos 80°, clip, gate θ>85°). Normal-direction variance `v_n = σ_r² cos²θ + r²σ_φ² sin²θ` (bounded) initialises splat covariances and floors their scale. Derivations in the manuscript's Appendix A. **Verified by hand; not a model of monocular error itself.**
- **Shared surface:** N_s Gaussians, **fixed budget** with MCMC relocation (3DGS-MCMC), depth-only fit with robust kernel + accumulated-alpha loss + anisotropy cap + depth-distortion/normal-consistency (2DGS-style). **Cross-fitting:** views split into K folds of neighbouring views; surface S_{-f} fitted on views outside fold f, rendered at views in f. A source never meets a surface fitted to it; a clustered faulty block is held out together.
- Held-out source: robust affine fit to the rendering (only non-affine deviation remains); compare on pixels with surface support α≥τ_α, valid, below the grazing gate.
- Statistic: normalised residual ρ_k against the rendered depth and variance, mean square T_k (chi-square gate) and mean log Bayes factor ℓ_k (evidence gate). Threshold calibrated on clean dev data to 5% per-view false exclusion (pixels are correlated).
- Gate outputs `g_{m,i}∈{0,1}`; **"no consensus"** if the surface does not support the view (supported fraction < s_min), M<3, or every source is rejected ⇒ view depth unused. Optional CUSUM for estimator-level exclusion.
- **Cost:** pairwise `C_pair ≈ N·d̄·P·c_w`; surface `C_surf ≈ ((K+1)·n_it + 1)·N·P·c_r`; break-even `d̄* ≈ ((K+1)·n_it + 1)·c_r/c_w`. **Linear in N for fixed K, N_s, n_it, but with a large constant: the pairwise gate is cheaper on sparse view graphs.** The pairwise gate (cross-view warp + set-based admission, entropy used only to select among admissible sets) is kept as the comparison (manuscript Appendix C).
- Caveats to test (H3, H6, breakdown): provisional poses corrupt both the surface and the gate; faulty views in *other* folds or dominating a region's coverage can make the surface follow the fault (masking); a depth-only surface is weakly constrained (floaters).

---

## 4. Baselines and reference points (configs/baselines/)

| Family | Methods | Registry name (stub) |
|---|---|---|
| Robust kernels (no gate) | Huber, Cauchy, Geman–McClure via GNC, switchable constraints, DCS | `robust_kernel_pgo` |
| Pixel filter | COLMAP geometric consistency (numpy test exists: `consistency_voting`) | `colmap_consistency_filter` |
| Plain voting | per-pixel median / majority across ensemble and neighbours (`median_fusion` exists) | `plain_voting` |
| Floor / bound / context | no gate; oracle gate; estimate-and-correct | `no_gate`, `oracle_gate`, `estimate_and_correct` |
| Gate variants | shared surface vs pairwise consensus × chi-square vs evidence statistic; ablations: no cross-fitting (joint fit), adaptive densification at matched size, exact leave-one-view-out refit (expensive reference) | `view_gate` (params `consensus`, `statistic`, `cross_fit`, `densification`) |
| Reference points | MASt3R-SfM, VGGT, COLMAP SfM | `mast3r_sfm`, `vggt`, `colmap_sfm` |

Same images, matches, ensemble, fault realisations and solver for all; thresholds tuned on dev scenes; tuning budget reported; re-implementations labelled. Reference points do not see injected depth faults (they ignore our depth) — context only. **All are stubs** (`src/sensorfault/baselines/gate_testbed.py`); nothing in the test bed is implemented yet.

---

## 5. Hypotheses (pre-register at week 1; mirrored in configs/h*.yaml and the manuscript)

- **H1 (gate vs the three baselines).** M≥3, one estimator corrupted on a fraction of views: gated pipeline cuts ATE and fused-surface error ≥25% vs the best tuned of {kernels, COLMAP filter, voting}, within 20% of oracle; advantage concentrated in clustered/few-neighbour faulty views and smooth faults; ties on sparse outliers. *Falsified if* the paired CI includes 0 in those regimes.
- **H2 (evidence vs chi-square).** At fixed 5% false exclusion, ≥10 points more power only where heteroscedastic noise, grazing, occlusion or non-Gaussian residuals break chi-square; ties under idealised Gaussian noise. *Falsified if* no gain anywhere → pivot P1.
- **H3 (cross-fitting, alignment, support).** Surface fitted *without* the source's view + scale/shift fit + support mask keep clean false exclusion ≤5%; fitting the surface to all views (incl. the source under test) lowers power ≥10 points (surface absorbs the fault); no support mask at least doubles false exclusion.
- **H4 (placement, complementarity).** Gating before PGO beats the same statistic after PGO; gate + robust kernel ≥ either alone.
- **H5 (clean cost, reference points).** Price of gating on clean data ≤5%; on natural failures the test bed narrows the gap to MASt3R-SfM/VGGT (context only).
- **H6 (shared surface vs pairwise checks).** (a) Efficiency: surface-gate time grows ~linearly in N (log–log slope <1.2) vs pairwise ∝ N·d̄ (slope >1.5 on dense graphs); ≥2× faster for N≥100 at d̄≥10; pairwise cheaper on sparse graphs (report break-even). (b) Robustness: power within 5 points of pairwise (non-inferior) in the H1 regimes; K∈{3,4,6} within 3 points of exact leave-one-view-out refit. (c) Fixed budget: adaptive densification at matched final size loses ≥10 points (surface adds a layer fitting the faulty source); sweet spot in N_s. *Falsified if* (a) speed-up <2× at dense overlap or slopes do not separate, (b) pairwise >5 points better, or (c) adaptive within 5 points of fixed.
- **Breakdown (honest negatives).** Common-mode estimator failure; M=2 (no consensus); sub-noise faults; sparse outliers (baselines win/tie); low overlap (surface cannot support the view → no consensus); **faulty views dominating a region's coverage (surface follows the fault)**; poor provisional poses.

Mapping from design v1: old H1 (sensor-level vs per-pixel) → new H1; old H2 → H2; old H3 (visibility/grazing) → H3 (+ alignment); old H4 (fixed-N splats) → **returns as H6c** (fixed budget vs adaptive densification, inside the shared-surface hypothesis); placement is the new H4; old H5 (learned fusion) → reframed as reference points.

---

## 6. Data and fault protocol (configs/datasets/, configs/faults/view_fault_grid.yaml)

| Dataset | Role | Notes |
|---|---|---|
| ETH3D | primary real | multi-view images, laser-scan GT depth, calibrated poses; natural estimator failures labelled against laser depth. **[UNVERIFIED]** scene names in base.yaml, licence, download script not written. |
| ScanNet++ | secondary | laser-scanned geometry + registered images; reflective/transparent surfaces; access terms to accept. |
| Replica | controlled synthetic | exact depth, vary overlap/fault parameters; research-only; release generator, not renderings. |
| TUM RGB-D, Tanks & Temples | metric definitions | ATE/RPE and reconstruction metrics. |
| H6 scaling study | view counts N∈{25,50,100,200}, mean co-visibility degree d̄∈{2,5,10,20} | Replica renderings (trajectory sets N and overlap), subsampled ScanNet++ sequences, ETH3D multi-camera videos. **[UNVERIFIED]** views per scene and achievable d̄ — check before fixing the grid. |

Injected view-level faults on one estimator: smooth warp (inverse depth, amplitude ×{0.25,0.5,1,2} std), region failure (20–40% area), non-affine distortion (γ∈{0.8,0.9,1.1,1.25}); fraction of views p∈{10,20,40,60}%, scattered vs clustered, estimator-level; 1–2 faulted estimators per view, independent vs identical (common mode); controls: sparse outliers, clean. Seeds: `SeedSequence([master, hash(source id), fault id])`, ≥5 masters. **Fault injectors for v2 are not implemented** (v1 injectors in `src/sensorfault/faults/` cover depth bias/extrinsic drift on physical sensors).

Metrics: ATE (Sim(3)), RPE, pose AUC; fused-surface accuracy/completeness/F-score; depth AbsRel/δ1 after alignment; gate precision/recall, false exclusion on clean data, power at FER 5%, elimination-curve AUC; per-stage runtime/memory. Scene-level bootstrap; paired Wilcoxon.

Compute risk: five estimators plus MASt3R-SfM/VGGT on 8 GB — cache depth maps offline, run reference points on subsets, check each model's memory **[UNVERIFIED]**.

---

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | Re-read the Sensors PDF equations; verify datasets/licences; ensemble weights licences; prereg tag | M1 |
| 2 | Estimator runner; cache ensemble depth on ETH3D dev scenes; per-estimator error vs laser | M2 |
| 3 | Matches; geometry-only graph (provisional poses); metrics harness | M3 |
| 4 | Depth-factor PGO; baselines: kernels, COLMAP filter, plain voting | M4 |
| 5 | View-level fault injectors; depth-only fixed-N splat fit (3DGS-MCMC/2DGS-style) with cross-fitting; chi-square gate (surface + pairwise) | M5 |
| **6** | **Gate 1:** H1 on dev scenes, 3 fault classes | go / pivot |
| 7 | Evidence gate; H2; H3 (cross-fitting, support, exact leave-one-view-out reference) | |
| 8 | Sweeps; H4 placement; **H6 scaling vs N and d̄ (break-even), fixed N vs adaptive**; re-gate pass | |
| 9 | Breakdown scenarios; **Gate 2** | honest negatives |
| 10 | Natural failures; reference points (MASt3R-SfM, VGGT, COLMAP SfM) | |
| 11 | ScanNet++ / Replica controlled sweeps (cut if behind) | |
| 12 | Gaps, estimate-and-correct, runtime table | |
| 13 | **Experiment freeze** | tag |
| 14–16 | Draft, figures, appendix, reproducibility package, AI-use disclosure | submission-ready |
| 17–18 | Buffer | |

**Gate 1:** go if the gate beats all three baselines in the H1 regimes. Pivot P1: evidence ≈ chi-square → the gate uses the chi-square statistic; entropy relation becomes an equivalence result. Pivot P2: the gate does not beat the best baseline → analysis paper ("when do pixel filters and kernels suffice?").
**Cut order:** ScanNet++/Replica sweeps → estimator-level CUSUM → exact leave-one-view-out reference → estimate-and-correct → optional estimators. Pivot P3: if the surface gate is not cheaper or is less reliable than the pairwise gate (H6 falsified), keep the pairwise gate as the method and report the surface as a negative result. **Never cut:** breakdown scenarios, clean-data cost (H5), the chi-square gate, the Eq. 14/Alg. 1 fix.

---

## 8. Paper outline (follows the author's own papers: Sensors 23(23):9309 and Sensors 24(23):7744)

| Section | Follows | Says |
|---|---|---|
| I Introduction, with A Relevant Literature, B Challenges (C1–C6), C Baseline Study and Fundamental Differences, D Summary of Contributions (mapped to challenges), E Organization | 2023 (1.1 limitations, 1.2 baseline study, 1.3 contributions, 1.4 organization) and 2024 (1.1 literature, 1.2 challenges list, 1.3 contributions mapped to challenges) | monocular depth fails coherently per view; where each method decides; gate is the contribution |
| II Method: A Definitions and Notation, B Forward Problem, C Inverse Problem (PGO), D View-Level Gate (requirements, consensus, corrected admission, output), E Test Bed | 2023 (Method with notation and BSE) and 2024 (forward/inverse problem) | the problem is set up inside the method, as in both papers |
| III Experiments: A Experimental Setup (data, faults, baselines, metrics), B Hypotheses, C Implementation | 2023 (3.1 Setup, 3.2 Implementation) | what is run and what would falsify it |
| IV Results | both | per hypothesis; placeholders |
| V Conclusions and Future Work (incl. Limitations) | 2023 | no separate Discussion section, as in both papers |
| Appendices | 2023 (theorems in appendix) | variance derivation; proof of Proposition 1; pairwise gate (comparison variant, Appendix C) |

Back matter in the author's papers (MDPI): author contributions, IRB, data availability, conflicts. For IEEE Access: Data and Code Availability (TODO), Acknowledgment with the AI-use disclosure, biography.

---

## 9. IEEE Access facts (verified on official pages 2026-11-13)
Verified on official pages: gold OA; APC $2,160 (+ taxes; 5%/20% member discounts only); no page limit, <20 pp recommended; single-anonymised, ≥2 reviews, ~4–6 weeks submission-to-publication, acceptance ~20%; reviewers judge technical soundness, not necessarily high novelty; no formal data/code statement required; preprints allowed (arXiv named in IEEE-wide policy, not on Access pages); AI-assisted writing must be disclosed in Acknowledgments; fabricated references → rejection; public ORCID required.
Not verified / yours to resolve: Virginia Tech–IEEE read-and-publish agreement (ask VT libraries; VT OASF announced closed Jul 2025); current IEEE copyright-form (eCF) wording and any publication-approval obligations that apply to you — check the live form. No Access conflict-of-interest page found. Not legal advice. Comparable-paper titles came from catalogue metadata only.

---

---

## 10. Open items
- [ ] Re-read the Sensors 23(23):9309 PDF: Alg. 1 inequality directions, entropy normalisation (author to confirm).
- [ ] Author to confirm the reading of "partly end-to-end".
- [ ] Verify ETH3D scene names/licence, ScanNet++ terms, each estimator's weights licence and GPU memory.
- [ ] Verify the number of views per scene and the achievable co-visibility degree in ETH3D / ScanNet++ for the H6 scaling grid.
- [ ] Literature pass: depth-prior filtering in SfM/SLAM; uncertainty-aware monocular depth in pose graphs.
- [ ] Implement: view-level fault injectors; pixel-wise gate; test bed pipeline; baselines; `data` download scripts for ETH3D.
- [ ] Choose and cite the matcher, solver and splat rasteriser; fix N_s, n_it, τ_α, s_min on dev scenes.
- [ ] Implement the depth-only fixed-N splat fit with cross-fitting and measure H6 early (week 5–8); check 8 GB memory for K+1 fits on ETH3D scenes **[UNVERIFIED]**.
- [ ] Affiliation / funding / AI-use disclosure in the manuscript (author's decision).

## 11. Change log
- 2026-11-13: v2 design. Gate is the contribution, pipeline the test bed; splat surface and physical depth sensors dropped; configs for v1 moved to `configs/legacy_sensor_faults/`; new configs, registry stubs and tests added.
- 2026-11-14: v2.1. Shared Gaussian-splat surface restored as the gate's consensus (cross-fitted, fixed budget, doubles as fused output); pairwise gate kept as comparison; H3 reworded (cross-fitting); H6 added (efficiency, robustness, fixed budget; old fixed-N idea returns as H6c); new breakdown scenario; gate variants and configs regenerated.
