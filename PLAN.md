# View-level fault rejection for depth-supervised Gaussian splatting — research plan (design v3)

Target: Elsevier `elsarticle` preprint first, then a subscription (no-APC) journal to be chosen; no pay-to-publish venue; solo author, 14–18 evening weeks (~10 h/wk), one 8 GB consumer GPU.
Title (fixed): *View-Level Fault Rejection for Monocular-Depth Reconstruction via a Shared Gaussian Surface*.
Status: v3 of 2026-11-14 (pivot after an external critique). No experiments yet. Items marked **[UNVERIFIED]** are not yet checked.
Hygiene: public datasets, public pretrained models and public code only; nothing from any employer.

---

## 1. Framing

- **Faults are regional and appearance-driven** (reflections and glass, textureless walls), many σ inside a connected region of a view. **No sub-noise claims.** Whole-view affine errors are absorbed by the per-view alignment and are not faults.
- **The gate must beat cheap baselines on these faults**: per-view aligned RMS, depth-model confidence, robust depth losses, RobustNeRF, SpotLessSplats.
- **3DGS camp:** 2DGS-style surfels rendered with gsplat into depth and variance. **Colour is a separate appearance block that never updates geometry**; colour reaches geometry only through the depth model. **The gate is depth-only.**
- **What is classical:** the test (Baarda data snooping / Teunissen DIA), solution separation = leave-one-out innovation (Joerger 2014), surfel fusion (Keller 2013), per-view alignment (MonoSDF).
- **What is new (claim):** every Gaussian carries an uncertainty that renders, used for fault testing with exact leave-view-out. Compare with FisherRF (Fisher information, view selection) and Bayes' Rays (post-hoc perturbation-field uncertainty).

## 2. Method

- **Surfel shape** (rendering only): μ, R = [t1 t2 n], scales s, opacity.
- **Surfel uncertainty** (separate): 9×9 information Λ_k over θ = [δμ (3, world), δφ (3, tangent-space/body rotation error, R ← R exp([δφ]×)), δ log s (3)].
- **Rendered sub-block:** normal offset ν = nᵀδμ and tilt τ1 = δφ2, τ2 = −δφ1, i.e. (ν, τ) = G θ. Ray-surfel z-depth d = nᵀ(μ−c)/nᵀw; ∂d/∂(ν,τ) = [1, −t1ᵀ(x−μ), −t2ᵀ(x−μ)]/nᵀw; J = (∂d/∂(ν,τ)) G (scale columns 0; scales keep their prior).
- **Rendering:** D(u) = Σ_k w_uk d_k (normalised alpha weights from gsplat); V(u) = Σ_k w_uk² J Λ_k⁻¹ Jᵀ.
- **Bayesian fusion on rendering:** view i adds Λ_k += Σ_u a² (w J)ᵀ(w J)/σ_u², η_k += Σ_u a (w J)ᵀ e_u/σ_u² (alpha-weighted rasteriser Jacobians, linearised at the fitted surfels). No correspondences.
- **Leave-view-out:** subtract view j's (Λ, η); exact for the linearised model.
- **Gate:** render the other views' surface into view j; drop surfels without other-view normal-offset information ≥ λ_min (**the tested view spawns no primitives**); pixels with accumulated opacity < τ_W are **untestable**; align (a_j, b_j); residual r = d̂ − (aD + b), Σ_r = DCD + a² diag V; energy statistic and region scan over appearance segments; 5% per-view false alarm calibrated on clean dev views; keep / exclude / untestable; scene-level alarm if >½ excluded; refit without excluded views.
- **Limitation (stated):** independent per-Gaussian blocks ignore cross-surfel, within-view and shared-pose correlations, so rendered variance is overconfident (H5; ablation: deflate view information by n_eff/n).
- Code: `src/sensorfault/eval/gaussian_uncertainty.py` (tests: `tests/core/test_gaussian_uncertainty.py` — Jacobian vs finite differences, sub-block map, exact leave-view-out, posterior vs Monte Carlo, calibrated z when blocks are exact, overconfidence with overlapping footprints and correlated noise, regional fault detection, no spawning, untestable pixels). Detection theory: `src/sensorfault/eval/detection_theory.py`. **Still to write:** a vectorised/GPU version on top of gsplat's per-pixel weights.

## 3. Measurement model and the first task

d̂ = a·d + b + ε, Σ_ε = D C D, D = diag σ(range, texture, grazing), C = (1−ν)K_ℓ + νI.
**Weeks 1–2, before anything else (H1):** on ETH3D dev scenes with Depth Pro, fit σ(·), ℓ, ν and tails of the aligned residuals against laser depth; characterise natural regional failures (area fraction, magnitude in σ, appearance class) and their harm to the fused surface; measure cross-view repetition of errors (H2). This is Figure 1. Go/no-go on H1.
Confidence outputs (public READMEs, 2026-11-14): Metric3D v2 (ViT) and UniDepthV2 yes; Depth Pro no; MoGe mask only; Depth Anything V2, Marigold none.

## 4. Hypotheses (configs/h*.yaml, mirrored in the manuscript)

- **H1 (residuals first; go/no-go).** Go if ≥10% of views carry a failure covering ≥2% of pixels at >3σ and keeping them costs ≥2 F-score points; no-go (analysis paper) below 5% or 1 point.
- **H2 (repetition across views).** Cross-view error correlation at the same laser point: ≥0.5 textureless, ≤0.3 reflective, ≤0.2 textured. Predicts a blind spot on textureless walls.
- **H3 (detection vs cheap baselines).** At 5% FA, on reflective-region failures, gate beats the best cheap/robust baseline by ≥15 points; no gain predicted on textureless.
- **H4 (reconstruction).** Depth-supervised splatting on ETH3D and ScanNet++: gate improves held-out AbsRel and F-score by ≥10% relative on failure scenes vs no gate and the best robust-splatting baseline; ≤2% change on clean scenes.
- **H5 (calibration).** Raw mean T/(n−2) > 1.2 on clean views (overconfidence); after deflation or threshold calibration, FA in [0.5α, 1.5α] on test scenes.
- **H6 (ablations).** Joint surface −10 pts; keeping tested-view-only surfels −20 pts on large regions; unwhitened −5 pts; rendered 9×9 ≥ FisherRF-style diagonal Fisher and Bayes' Rays-style uncertainty.
- **Breakdown:** common-mode repetition, low overlap (untestable), faulty views dominating a region, misspecified measurement model, linearisation limit.

## 5. Baselines (configs/baselines/)

| Family | Methods | Registry |
|---|---|---|
| Robust splatting | SpotLessSplats, RobustNeRF loss, Huber / Geman–McClure depth loss | `spotless_splats`, `robustnerf_loss`, `robust_depth_loss` |
| Cheap view tests | per-view aligned RMS vs other views' render; depth-model confidence | `aligned_rms_view_test`, `depth_confidence_score` |
| Bounds | no gate; oracle gate | `no_gate`, `oracle_gate` |
| Gate and ablations | full; joint; keep unsupported; unwhitened; deflated; FisherRF-style; Bayes' Rays-style | `view_gate` |

## 6. Data and faults (configs/datasets/, configs/faults/view_fault_grid.yaml)

ETH3D (primary; residual measurement; laser depth + calibrated poses) and ScanNet++ (secondary; terms to accept). Poses from the datasets; pose faults out of scope.
Natural failures labelled vs laser on connected regions (reflective/glass, textureless wall, other). Injected regional faults: offset, plane replacement, blurred copy, mirrored depth; 2–40% of a view; 2–16σ; on appearance masks or random superpixels; independent or repeated across views. Controls: fault-free; whole-view affine (absorbed).

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | ETH3D licence/scenes; Depth Pro licence and 8 GB memory; depth cache on dev scenes; prereg tag | M1 |
| 2 | **H1 residual measurement + natural-failure characterisation + harm; H2 repetition** (Figure 1) | **go / no-go** |
| 3 | gsplat 2DGS depth-supervised fit with the appearance block (stop-grad); metrics harness | M3 |
| 4 | Information accumulation from gsplat per-pixel weights (vectorised); leave-view-out render | M4 |
| 5 | Gate: whitening, energy + region scan, calibration (H5) | M5 |
| 6 | Cheap baselines (aligned RMS, depth confidence); fault injectors | |
| 7–8 | H3 detection; SpotLessSplats / RobustNeRF / robust depth losses | Gate 2 |
| 9–10 | H4 reconstruction on ETH3D, then ScanNet++ | |
| 11 | H6 ablations (FisherRF-style, Bayes' Rays-style); breakdown scenarios | |
| 12 | **Experiment freeze** | tag |
| 13–15 | Draft, figures, supplement, reproducibility package | preprint-ready |
| 16–18 | Buffer | |

**Cut order:** ScanNet++ → Bayes' Rays-style ablation → FisherRF-style ablation → injected-fault sweeps beyond the core grid. **Never cut:** H1, H3 against the cheap baselines, H5.

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
- [ ] **Weeks 1–2: H1 residual measurement and H2 repetition on ETH3D dev scenes** (go/no-go).
- [ ] Verify ETH3D scene names/licence, ScanNet++ terms, Depth Pro weights licence and 8 GB memory.
- [ ] Baarda (1968) is named but cited through Teunissen (2018); add a checked Baarda reference if one can be verified.
- [ ] Final literature pass: uncertainty-aware Gaussian splatting; depth-prior filtering.
- [ ] Vectorised/GPU gate on gsplat; fix λ_min, τ_W, Λ_0, segmentation on dev scenes.
- [ ] AI-use declaration and code/data statement in the manuscript (author's decision).

## 11. Change log
- 2026-11-13: v2 design (gate = contribution, pipeline = test bed).
- 2026-11-14: v2.1 shared splat surface restored; v2.2 one metric depth model per image, ensemble → ablation.
- 2026-11-14: **v2.3 integrity-monitoring framing (RAIM-style FDE).** Core argument made explicit (sub-noise per pixel, coherent per view; MDB ∝ 1/(f√n_eff)); pose faults added; soft down-weighting is a baseline; conditional protection level added (optional); hypotheses reorganised to H1–H10 (new H1 regime map, H8 pose faults, H9 protection level; ties/losses pre-registered where gating should not help); `detection_theory` module + tests; configs and tests regenerated; five RAIM/integrity references added.
- 2026-11-14: **v2.4 explicit measurement model.** Nominal d̂ = a·d + b + ε with scale/offset ambiguity, heteroscedastic spatially-correlated noise (σ(range, texture, grazing), ℓ, nugget ν), pose, intrinsics and surface uncertainty (jackknife over K cross-fitted surfaces) → Σ_r; tests on whitened, nuisance-projected residuals; n_eff = tr(C)²/tr(C²) computed; MDB, scan and protection level from Σ_r. **Correction:** a bias uniform over a whole view (f=1) or ∝σ(d) is an offset, absorbed by the alignment — not a fault, not detected, not harmful; the regime map now uses f ≤ 0.8 and an f=1 affine control (H1d); the signed-mean statistic is dropped in favour of a region scan; the earlier MDB table is withdrawn. New: calibration protocol and public-README check of which models output confidence; adequacy test in H6; misspecification breakdown; week-2 go/no-go is now measuring σ(·), ℓ, ν. Code: tested measurement-model theory (nugget correlation grid, Σ_r assembly, jackknife surface covariance, projected precision, κ, MDBs, n_eff, pose Jacobian, ellipsoid PL).
- 2026-11-14: **v2.5 surface reframe.** Depth maps only (colour dropped; the photometric-term ablation and future work removed); Gaussian planar primitives over an SDF; extent separated from uncertainty; closed-form forward (σ_h²=nᵀPn from depth, pose and intrinsics); information-form fusion with exact leave-view-out by subtraction (= solution separation = innovation); ray-plane backward render with alpha compositing and extent-only weights; Σ_surf = a²diag σ²(u) replaces the K-fold jackknife; projective association with soft weights; per-region posterior, protection level and view count; adaptive densification dropped, fixed-budget MCMC kept as an ablation only. H4 reworded (leave-view-out vs joint), H7 revised (closed form vs pairwise vs MCMC fixed budget; association/view order) — the request's "H6" for the shared-surface hypothesis is H7 in this numbering. New breakdown `association_failure`; configs `gate_surface_mcmc_fixed_budget`, `gate_surface_single_covariance`; code `surface_model.py` (12 tests) and `sigma_surf_diag`. Schedule: week 5 builder is closed form.
- 2026-11-14: **v2.6 venue and layout.** Elsevier `elsarticle` replaces the IEEE Access template (no pay-to-publish; preprint first); IEEE-specific items removed; affiliation Independent Researcher, Shelton, CT, no external funding; the "partly end-to-end" discussion and its open item dropped; the Algorithm 1 vs Eq. 14 item closed as a typo in the 2023 paper (follow Eq. 14); main text cut to 8 pages with related work, challenges, test bed, ensemble, full setup, planned tables/figures and derivations moved to a Supplementary Material; ablation A5 (literal Algorithm 1) removed.
- 2026-11-14: **v3 pivot.** All sub-noise fault claims dropped; faults are regional and appearance-driven (reflections, textureless walls) and the gate must beat cheap baselines. 3DGS camp: 2DGS surfels rendered with gsplat into depth and variance; colour is a separate appearance block that never updates geometry; the gate is depth-only. Each Gaussian keeps its shape covariance for rendering plus a separate 9×9 uncertainty (mean, tangent-space rotation error, log scales); the normal-offset + tilt sub-block is rendered. Bayesian fusion on rendering (alpha-weighted J^T R^-1 J), leave-view-out by subtraction, other views' surface rendered into view j, whitened residuals tested; no correspondences; unseen pixels untestable; tested view spawns nothing. Novelty restated (uncertainty that renders, used for fault testing) against FisherRF, Bayes' Rays, Baarda/Teunissen, Joerger 2014, surfel fusion, MonoSDF. Baselines: SpotLessSplats, RobustNeRF, robust depth losses, per-view aligned RMS, depth-model confidence; evaluation by depth-supervised splatting on ETH3D and ScanNet++ with and without the gate; ETH3D residual measurement first. Limitation stated: independent per-Gaussian blocks → overconfident rendered variance. New H2 tests repetition of errors across views in textureless regions. Removed: pose graph test bed, pose faults, RAIM protection level, pairwise gate, ensemble ablation, MASt3R-SfM/VGGT reference points, closed-form association surface (`surface_model.py`). New: `gaussian_uncertainty.py` + tests; configs v3 (H1–H6, 5 breakdowns, A0–A10); registry stubs for the new baselines.
