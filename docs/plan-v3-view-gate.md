> Superseded by design v4 (`PLAN.md`). Kept because the view gate described here is the depth-consensus baseline of the current paper, and E1 and the pose check were run under it.

# View-level fault rejection for depth-supervised Gaussian splatting — research plan (design v3.4)

Target: Elsevier `elsarticle` preprint first, then a subscription (no-APC) journal to be chosen; no pay-to-publish venue; solo author, 14–18 evening weeks (~10 h/wk), one 8 GB consumer GPU.
Title (fixed): *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*.
Status: v3.4 of 2026-11-15 (pivot after an external critique, the decisions of `docs/design-3dgs-bayes.md` §7, then the outer loop, then the aerial setting, spline poses and leave-block-out, then off-the-shelf depth only and the pose check). **Reference design: `docs/design-3dgs-bayes.md`** (§8, the per-view bias field, is background for a separate paper and stays out of this one beyond a one-line future-work pointer). No experiments yet. Items marked **[UNVERIFIED]** are not yet checked.
Hygiene: public datasets, public pretrained models and public code only; nothing from any employer.

---

## 1. Framing

- **Contribution: the gate.** The representation is standard 2DGS surfels augmented with information blocks — not new. Position it as sparse, with correspondence-free fusion and testing, unlike dense TSDF/voxels, NeRF fields or MVS cost volumes; geometry is depth-only, so reflectance cannot corrupt it.
- **Setting: aerial survey imagery** (UAV nadir strips, GNSS/IMU or exterior orientation). Main real set UseGeo; H3D pending; MatrixCity synthetic for exact-truth fault injection; ETH3D a small sanity check; ScanNet++ dropped. Monocular depth models are trained mostly near the ground, so nadir views from altitude are a different failure regime (H1 measures it).
- **Lighting:** light-source estimation and relighting are future work only (one line; varying-illumination aerial set as pointer).
- **Depth source: off-the-shelf released models only** (Depth Pro default; Metric3D v2 and UniDepthV2 for the confidence baseline and the altitude comparison). No unreleased or in-house depth model in this paper. The altitude failure comparison (ground-trained models on nadir views from altitude vs near-ground views) is a results item.
- **Downloads:** nothing large without the author's explicit yes (a single MatrixCity aerial block tar is about 15.8 GB; UrbanScene3D is 1.43 TB).

- **Faults are regional and appearance-driven** (reflections and glass, textureless walls), many σ inside a connected region of a view. **No sub-noise claims.** Whole-view affine errors are absorbed by the per-view alignment and are not faults.
- **The gate must beat cheap baselines on these faults**: per-view aligned RMS, depth-model confidence, robust depth losses, RobustNeRF, SpotLessSplats.
- **3DGS camp:** 2DGS-style surfels rendered with gsplat into depth and variance. **Colour is a separate appearance block that may refine poses but never updates map geometry or opacity**; colour reaches map geometry only through the depth model. **The gate is depth-only.**
- **What is classical:** the test (Baarda data snooping / Teunissen DIA), solution separation = leave-one-out innovation (Joerger 2014), surfel fusion (Keller 2013), per-view alignment (MonoSDF).
- **What is new (claim):** every Gaussian carries an uncertainty that renders, used for fault testing with exact leave-view-out. Compare with FisherRF (Fisher information, view selection) and Bayes' Rays (post-hoc perturbation-field uncertainty).

## 2. Method (reference: docs/design-3dgs-bayes.md §1–4, 9, 10)

- **Surfel shape** (rendering only): μ, R = [t1 t2 n], scales s, opacity α.
- **Surfel uncertainty** (separate): **10×10** information Λ_k over θ = [δμ (3, world), δφ (3, tangent-space/body rotation error, R ← R exp([δφ]×)), δ log s (3), δ logit α (1)]. Opacity is the 10th coordinate, **trained in a warm-up and then frozen**; it is not re-estimated in the leave-out and is used for the support rule.
- **Rendered sub-block:** normal offset ν = nᵀδμ and tilt τ1 = δφ2, τ2 = −δφ1. Ray-surfel z-depth d = nᵀ(μ−c)/nᵀw; ∂d/∂(ν,τ) = [1, −t1ᵀ(x−μ), −t2ᵀ(x−μ)]/nᵀw. Opacity enters through the compositing weights (∂D/∂logit α incl. transmittance, from the rasteriser backward pass). Scales keep their prior.
- **Rendering:** D(u) = Σ_k w_uk d_k (gsplat); V(u) = Σ_k w_uk² J P_k Jᵀ with P_k the inverse geometric 9×9 block.
- **Fusion:** Laplace at the MAP, Λ_k = Λ_0 + Σ_i Σ_u a² gᵀg/(γ² σ_u²), g = alpha-weighted rasteriser Jacobian. No correspondences.
- **Leave-view-out:** Λ_{−j} = Λ − Λ_j plus one Newton step μ_{−j} = μ + Λ_{−j}⁻¹ ∇L_j on the 9 geometric coordinates (infinitesimal jackknife; one backward pass of view j; exact for the linear-Gaussian model).
- **Noise:** Σ_ε = γ² D C D. Shape (σ(·), ℓ, ν) calibrated offline on dev; **one free noise variable, the scene temperature γ**, fitted leave-view-out (γ̂²_{−j}: median_{i≠j} T_i/q_0.5(χ²_{m_i}) = 1). **No per-view or per-pixel noise parameters.** γ̂ above its dev range → scene-level alarm.
- **Shared bias:** distance term d → d + β d², β shared by all views, least squares; leave-view-out by subtracting view j's normal equations. Grazing/texture terms only if dev residuals show them. Per-view bias field: separate paper.
- **Poses (v3.3):** input is posed images. Trajectory = uniform cubic B-spline in time (positions standard basis; rotations cumulative on SO(3)), fitted to GNSS/IMU or dataset poses with a second-difference prior, then control points refined by differentiable PGO with GNSS/IMU prior + smoothness + render-and-compare depth terms (optionally photometric render-and-compare); **no feature correspondences**. Tested view: its own render-and-compare term is subtracted before use; Σ_q = b(t_j) Σ_ctrl b(t_j)ᵀ enters Σ_r. Ablation: perturbed starting poses. ETH3D: registered poses as given. Pose faults out of scope. Code: `src/sensorfault/eval/trajectory_spline.py` (tests `tests/core/test_trajectory_spline.py`).
- **Pose check (v3.4, H7):** after the gate has run, perturb poses of chosen views (isolated, a block, the flagged view) by known rotations {0.05…2°} and translations {1…50 GSD}; check (a) whether the gate flags them before refinement (own depth excluded from the tested pose) and (b) whether spline + PGO + render-and-compare pulls them back; report detection rate and recovery error vs perturbation size. Pose faults are not a target of the gate; this is a controlled check. Helpers `perturb_pose`, `pose_error`, `recovery_curve` (tests: exact known perturbation, spline refit pulls a perturbed frame back).
- **Leave-block-out (v3.3):** cavity for view j removes a block of B neighbouring frames of the same strip (centred, shifted inwards at strip ends, never across strips) by subtracting the block's summed information + one Newton step (exact for the linear model). Motivation: neighbouring frames repeat the same error (H2) and would mask it. Only view j is tested; Holm family unchanged, M = N·R_max (block-level testing would be ⌈N/B⌉·R_max; not used). In the loop a rejected view is dropped alone. Cost: weaker cavity (higher MDB, more untestable pixels). B from the sweep {1,3,5,9} on dev against H2's correlation decay; B=1 is the ablation. Code: `block_for_view`, `block_information`, `leave_block_out_newton` in `gaussian_uncertainty.py` (tests: strip windows, exact block subtraction = refit = sequential subtraction, weaker cavity).
- **Gate:** render the other views' surface into view j; drop surfels without other-view normal-offset information ≥ λ_min **or** with Λ_j[α,α]/Λ[α,α] > ρ = ½ (**the tested view spawns nothing**); pixels with accumulated opacity < τ_W are **untestable**; align (a_j, b_j); whiten with γ̂_{−j}; energy statistic (n_eff dof; threshold absorbs the per-view heterogeneity ω) and region scan over appearance segments; exclusion by the outer loop below; scene-level alarm if >½ excluded.
- **Algorithm "train, freeze, fuse, test":** cached depth → warm-up (μ, R, s, α on depth; (a,b) closed form; β by LS; densify only with ≥2 supporting views; prune) → freeze opacity, stop densify/prune → fit geometry and poses to convergence, colour on frozen geometry → Laplace Λ → gate loop.
- **Outer loop (v3.2):** rounds of train (opacity unfrozen again, densify/prune on, warm-started from the previous round) → freeze → fuse → test. Default: drop the single most significant Holm-rejected view per round (swamping protection), no readmission (monotone, terminates), round cap R_max = 10; stop when no view is rejected, at the cap, or at the scene-level alarm. **Holm across rounds:** family M = N·R_max (every view in every round, re-tests and readmission tests included); the (k+1)-th rejection overall uses α/(M−k); step-down within a round; overall level calibrated by clean-run permutation of the whole loop (p-values change between rounds). Removes the warm-up pruning risk for caught views (they stop shaping the geometry; the next round re-densifies without them). Risks: masking (resolved over rounds), swamping (one view per round), cap reached (alarm). Ablations: loop off (single pass), drop all rejected per round, down-weight instead of drop, readmission, cap 3. Framing: Baarda-style iterative data snooping and hard EM (gate = E-step, train = M-step), EP cavity within each round — known ideas, not contributions. Fusion is not differentiated; gradients run only inside the train step. Code: `src/sensorfault/eval/gate_loop.py`, tests `tests/core/test_gate_loop.py` (Holm counting, masking, swamping, readmission, down-weighting, alarm, null FWER ≤ α by simulation).
- **Limitation (stated; no deeper fix yet):** independent per-Gaussian blocks ignore cross-surfel, within-view and shared-pose correlations, so the rendered variance is overconfident (H5). Ablations: deflate view information by n_eff/n; jackknife inflation.
- Code: `src/sensorfault/eval/gaussian_uncertainty.py` (10-dim block, opacity Jacobian, Newton leave-out, opacity support rule, shared distance bias) and `src/sensorfault/eval/detection_theory.py` (scene temperature, heterogeneity-calibrated energy threshold). Tests: `tests/core/test_gaussian_uncertainty.py`, `tests/core/test_detection_theory.py`. **Still to write:** a vectorised/GPU version on top of gsplat's backward pass.

## 3. Measurement model and the first task

d̂_i = a_i (d + β d²) + b_i + ε, Σ_ε = γ² D C D, D = diag σ(range, texture, grazing), C = (1−ν)K_ℓ + νI.
**Weeks 1–2, before anything else (H1):** on UseGeo (LiDAR depth maps) and ETH3D dev scenes with Depth Pro, nadir-from-altitude and near-ground views separately, fit σ(·), ℓ, ν, tails, β, the γ range and the per-view heterogeneity ω of the aligned residuals against laser depth; characterise natural regional failures (area fraction, magnitude in σ, appearance class) and their harm to the fused surface; measure cross-view repetition of errors (H2). This is Figure 1. Go/no-go on H1. If ω ≳ 0.05, lead with the region scan (energy MDB grows ×1.64 at ω = 0.05, ×2.43 at 0.10).
Confidence outputs (public READMEs, 2026-11-14): Metric3D v2 (ViT) and UniDepthV2 yes; Depth Pro no; MoGe mask only; Depth Anything V2, Marigold none.

## 4. Hypotheses (configs/h*.yaml, mirrored in the manuscript)

- **H1 (residuals first; go/no-go).** Also measures β, the γ range and ω. Go if ≥10% of views carry a failure covering ≥2% of pixels at >3σ and keeping them costs ≥2 F-score points; no-go (analysis paper) below 5% or 1 point.
- **H2 (repetition across views).** Cross-view error correlation at the same reference point: ≥0.5 textureless, ≤0.3 specular, ≤0.2 textured, decaying with frame separation along the strip (sets B). Predicts a blind spot for errors repeated beyond the block.
- **H3 (detection vs cheap baselines).** At 5% FA, on reflective-region failures, gate beats the best cheap/robust baseline by ≥15 points; no gain predicted on textureless.
- **H4 (reconstruction).** Depth-supervised splatting on UseGeo and ETH3D (natural failures) and MatrixCity (injected faults): gate improves held-out AbsRel and F-score by ≥10% relative on failure scenes vs no gate and the best robust-splatting baseline; ≤2% change on clean scenes.
- **H5 (calibration).** Raw mean T/m > 1.2 on clean views (overconfidence of independent blocks); with the leave-view-out scene temperature and the ω-calibrated energy threshold, FA in [0.5α, 1.5α] on test scenes; n_eff/n deflation and jackknife inflation are ablations.
- **H6 (ablations).** Joint surface −10 pts; keeping tested-view-only surfels −20 pts on large regions; unwhitened −5 pts; rendered 10×10 ≥ FisherRF-style diagonal Fisher and Bayes' Rays-style uncertainty; no temperature / no shared bias raise clean FA; fold-wise opacity warm-up changes FA and power by < 2 pts. Outer loop vs single pass: ≥10 pts more detections when ≥2 faulty views share a region, clean FA within band; all-rejected policy swamps; readmission changes < 2 pts. Leave-block-out (B from {1,3,5,9}) ≥10 pts over B=1 when neighbours repeat the error; perturbed starting poses <5 pts.
- **H7 (pose perturbation check).** (a) gate flags ≥80% of views perturbed by ≥0.5° or ≥10 GSD before refinement; (b) refinement removes ≥80% of the perturbation up to 1° / 25 GSD; recovery error vs size reported, including failures.
- **Breakdown:** common-mode repetition, low overlap (untestable), faulty views dominating a region, misspecified measurement model, linearisation limit.

## 5. Baselines (configs/baselines/)

| Family | Methods | Registry |
|---|---|---|
| Robust splatting | SpotLessSplats, RobustNeRF loss, Huber / Geman–McClure depth loss | `spotless_splats`, `robustnerf_loss`, `robust_depth_loss` |
| Cheap view tests | per-view aligned RMS vs other views' render; depth-model confidence | `aligned_rms_view_test`, `depth_confidence_score` |
| Bounds | no gate; oracle gate | `no_gate`, `oracle_gate` |
| Gate and ablations | full; joint; keep unsupported; unwhitened; deflated; jackknife-inflated; no temperature; no shared bias; fold-wise warm-up; loop off; drop all rejected; down-weight; readmission; FisherRF-style; Bayes' Rays-style | `view_gate` |

## 6. Data and faults (configs/datasets/, configs/faults/view_fault_grid.yaml)

Licences, checked against the actual pages/files on 2026-11-15 — **all non-commercial**, recorded in the data statement:
- UseGeo (main real): CC BY-NC-SA 4.0, stated in the README "License" section of github.com/3DOM-FBK/usegeo (no LICENSE file in the repo; the dataset site itself names none). Three sets of 8 strips (224/328/277 images).
- MatrixCity (synthetic): CC BY-NC 4.0 on the Hugging Face card BoDai/MatrixCity (not gated); depth, normals and transforms.json poses confirmed in the file list; one aerial test block ≈ 15.8 GB.
- H3D (pending): download link only after registration; data "as is"; no licence stated; pages describe LiDAR point clouds and textured meshes, images/poses not stated — treated as non-commercial research use until confirmed.
- UrbanScene3D (optional, one scene at most): non-commercial only, "AS IS", no dissemination of the data or altered variations; 1.43 TB in total.
- ETH3D (sanity check): licence still to check. Fallback: ISPRS Vaihingen (research only, no redistribution). Not used: Dortmund (125–350 GB), Mill-19 (no geometric reference), ScanNet++ (dropped).
Natural failures labelled vs LiDAR/laser on connected regions (specular glass/water, textureless roofs/roads/walls, other). Injected regional faults: offset, plane replacement, blurred copy, mirrored depth; 2–40% of a view; 2–16σ; on appearance masks or random superpixels; independent or repeated across views. Controls: fault-free; whole-view affine (absorbed).

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | UseGeo and ETH3D access/licence/scenes; Depth Pro licence and 8 GB memory; depth cache on dev scenes; prereg tag | M1 |
| 2 | **H1 residual measurement + natural-failure characterisation + harm; H2 repetition** (Figure 1) | **go / no-go** |
| 3 | gsplat 2DGS depth-supervised fit: warm-up, opacity freeze, appearance block (stop-grad into geometry/opacity, may refine poses); shared distance bias; metrics harness ; trajectory spline fit and render-and-compare pose refinement | M3 |
| 4 | Laplace 10×10 information from gsplat backward passes (vectorised); Newton leave-out and leave-block-out render; support rules | M4 |
| 5 | Gate: scene temperature, whitening, energy (ω threshold) + region scan, outer loop (Holm across rounds), calibration (H5) | M5 |
| 6 | Cheap baselines (aligned RMS, depth confidence); fault injectors | |
| 7–8 | H3 detection; SpotLessSplats / RobustNeRF / robust depth losses | Gate 2 |
| 9–10 | H4 reconstruction on UseGeo and MatrixCity, then ETH3D; block-size sweep | |
| 11 | H6 ablations (FisherRF-style, Bayes' Rays-style); breakdown scenarios | |
| 12 | **Experiment freeze** | tag |
| 13–15 | Draft, figures, supplement, reproducibility package | preprint-ready |
| 16–18 | Buffer | |

**Cut order:** H3D → Bayes' Rays-style ablation → FisherRF-style ablation → injected-fault sweeps beyond the core grid. **Never cut:** H1, H3 against the cheap baselines, H5.

## 8. Paper outline (elsarticle; main text ≤ 8 pages)

1 Introduction; 2 Related Work (short); 3 Method (setting and faults, surfels with an uncertainty that renders, fusion on rendering and leave-view-out, the view gate); 4 Experimental Design and Hypotheses; 5 Results; 6 Conclusions, Limitations, Future Work; Declarations; References; Supplement (related work, derivations, setup in full, planned tables and figures).

## 9. Venue and preprint plan
- Template: Elsevier `elsarticle` (TeX Live), `5p,twocolumn`; main text at most 8 pages, then declarations, references and a Supplementary Material appended to the same PDF (split into a separate file if the chosen journal requires it).
- No pay-to-publish venue: no APC / open-access fee. Preprint first (e.g. arXiv, once the results exist and the repositories are ready to be made public); choose a subscription-model Elsevier journal afterwards. **[UNVERIFIED]** which journal and its current policies: check its guide for authors before submitting.
- Elsevier asks for a declaration of generative-AI use and a data/code availability statement: both are open TODOs in `main.tex`.
- The IEEE Access template, biography and boilerplate were removed from the manuscript repo.

---

---

## 10. Open items
- [ ] **Weeks 1–2: H1 residual measurement and H2 repetition on UseGeo and ETH3D dev scenes** (go/no-go).
- [ ] Confirm ETH3D licence; ask the H3D maintainers about images, orientations and licence; pick at most one UrbanScene3D scene (check which real scenes have LiDAR and poses). **No large download without the author's yes.**
- [ ] Baarda (1968) is named but cited through Teunissen (2018); add a checked Baarda reference if one can be verified.
- [ ] Final literature pass: uncertainty-aware Gaussian splatting; depth-prior filtering.
- [ ] Vectorised/GPU gate on gsplat; fix λ_min, ρ, τ_W, Λ_0, warm-up length, γ_max and the segmentation on dev scenes.
- [ ] Holm (1979) is named for the step-down schedule but not yet cited (no DOI found); add a checked reference.
- [ ] AI-use declaration and code/data statement in the manuscript (author's decision).
