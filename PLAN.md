# Sensor-level fault elimination in a shared Gaussian-splat surface — research plan

Target: IEEE Access, solo author, 14–18 evening weeks (~10 h/wk), one consumer GPU.
Status: merged from 8 research agents on 2026-11-13. Items marked **[UNVERIFIED]** come from memory or abstract-only reads and must be checked before they appear in the paper.
Hygiene: public datasets, public sensor models and public code only.

---

## 1. Contribution, stated narrowly

> Consensus-render-based, whole-sensor (or whole-frame) exclusion before dense Gaussian-splat fusion, with a corrected, evidence-based admission test.

Do NOT claim "nobody drops sensors". Precedents exist (RAIM/FDE in navigation; Byzantine per-robot exclusion in Swarm-SLAM; whole-frame rejection in DROID-Splat; measurement-level rejection in ToF-Splatting). The novelty is the dense splat-consensus application and sensor-level attribution under coherent faults.

### Novelty check result (agent: novelty)
No paper found that tests whole sensors against a Gaussian-splat consensus. Search was ~12 web queries, abstract-level, no IEEE Xplore/Scholar. **Not a proof of absence.**

| Work | Verdict |
|---|---|
| BayesianGS-SLAM (arXiv 2609.24140, Sep 2026) | Single RGB-D sensor; uncertainty-weighted tracking + keyframe skipping. Low threat. Abstract page only seen by agent — your full read supersedes. |
| 3DGS-Calib (arXiv 2403.11577) | LiDAR-camera calibration; no fault detection. Low. |
| ToF-Splatting (arXiv 2504.16545) | Drops individual ToF measurements by residual quantile. **Med** — cite and contrast. |
| DROID-Splat (arXiv 2411.17660) | Whole-frame test vs rendered disparity, one sensor. **Med.** |
| SenFuNet (ECCV 2022) | Learned soft per-sensor weights. **Med** — reviewer will cite it; also a baseline. |
| arXiv 2306.11048 | Multi-sensor neural-implicit SLAM, pixel-wise uncertainty. Med. |
| YOGO (arXiv 2604.21400) | "availability-registration protocol for multi-sensor fusion". **Unresolved — read full text before committing to novelty.** |
| MiCaDe (arXiv 2611.04912) | LiDAR-camera miscalibration change detection, not splats. Low-Med. |

Still to do: IEEE Xplore/Scholar pass on "leave-one-out / sensor exclusion + TSDF/volumetric fusion" and "Gaussian splatting + sensor/modality dropout"; verify PSDF Fusion and SenFuNet grazing-angle claims (not checked); Ground-Fusion++ (arXiv 2507.08364) unchecked; ignore "LIT-GS 2606.20424" until verified.

---

## 2. Fix for Eq. 14 / Algorithm 1 (Ambarkutuk et al., Sensors 23(23):9309)

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

**Equivalence warning (agents: classical + design).** For the unnormalised-product surprisal, −log Z = ½νᵀS⁻¹ν + ½log det(2πS): a leave-one-out Mahalanobis test plus a volume term. Solution separation equals the leave-one-out innovation statistic when H is invertible. So under Gaussian residuals your criterion ≈ leave-one-out χ². The paper must show where it adds something (heteroscedastic noise, non-Gaussian/multimodal residuals, occlusion, grazing) or concede equivalence (pivot P1 below).

---

## 3. Per-ray variance and splat covariance (agent: covariance — **from memory, no web**)

- σ_r² + (rσ_φ tanθ)² is correct as the variance of the **depth residual along the ray**: dr/dθ = r tanθ exactly.
- It is NOT the 3D point error. In the ray frame the covariance is diag(σ_r², (rσ_φ)², (rσ_φ)²). Projected on the surface normal: σ_n² = σ_r²cos²θ + r²σ_φ²sin²θ, bounded as θ→90°.
  → Use the tanθ form to **weight depth residuals**; use σ_n / ray-frame form to **set splat covariance**.
- Grazing: linearisation fails when σ_φ tanθ ≳ 0.3. Clamp: tan²θ → sin²θ/(cos²θ+ε²), ε≈0.17; hard-gate θ > 85°. Add beam-footprint term σ_φ,eff² = σ_φ² + (κγ)². Floor/cap the variance.
- Stereo/structured light: σ_z = z²σ_d/(fb) → variance z⁴σ_d²/(fb)² (square the fb). ToF noise ∝ 1/amplitude; multipath/flying pixels are biases, not variance — mask them.
- Use of variance in the model: init + likelihood weight (½ r²/var) + soft floor on scale; not a fixed prior (posterior variance shrinks ~1/N).
- 3DGS-MCMC fixed N, depth-only: add accumulated-alpha mask loss (else opacity L1 collapses opacity), cap anisotropy, scale SGLD noise below σ_n, 2DGS depth-distortion + normal-consistency terms rescaled to metric units. LiDAR needs a non-pinhole rasteriser.
- References to verify: Kheradmand 2024 (3DGS-MCMC), Huang 2024 (2DGS), Khoshelham & Elberink 2012, Nguyen/Izadi/Lovell 2012, Lange & Seitz 2001, Matthies & Shafer 1987, Sarbolandi 2015, Glennie & Lichti 2010, Soudarissanane 2011. Constants marked [U] by the agent are unverified.

---

## 4. Methods and baselines

### Sensor-level elimination procedure
1. Joint fit (fixed-N MCMC splats, depth-only).
2. Visibility-aware residuals r = (D_s − D̂)/σ_s(z,θ) only where rendered alpha > τ and the first-return gate matches (occlusion handling).
3. Per-sensor statistics: leave-one-out χ² (warm-started LOO refit, ~200 iters, to counter masking), Δ-entropy (mixture-of-per-sensor-posteriors form), Moran's I residual autocorrelation (diagnostic only).
4. Exclude worst sensor if statistic exceeds a threshold calibrated on clean-run permutations (5% false-elimination); refit; iterate; keep ≥ 2–3 sensors.
5. Dynamics: per-sensor persistent residual rise (CUSUM) = sensor fault; localised, multi-sensor-visible, few-frame residual = dynamic object → mask.
6. Implement the Alg. 1 literal reading and the corrected reading as variants; pre-register the corrected one.

### Baselines — main benchmark: three families (updated)

Protocol for ALL methods: identical fault-injected depth maps, identical (possibly drifted) poses and intrinsics, identical view graph, identical per-pixel confidence (or none), same fault realisations/seeds. Thresholds tuned on dev scenes only, tuning budget reported, sweeps shown. Reimplementations are labelled as such and released. Offline baselines (OpenMVS) are accuracy baselines only; report latency for every method and state which miss the < 100 ms/frame target.

**Family A — OpenMVS depth-map fusion (primary voting baseline)** (agent read `develop` source; nothing built or run)
- Fusion = per-pixel consistency voting: back-project each depth pixel, project into up to 32 neighbour views, keep if relative depth (default 1%) and normal (25 deg) agree in >= `--number-views-fuse` views (default 2), zero neighbours that project in front and disagree (free-space filter), confidence-weighted average.
- Variants to report: `--fusion-filter 0` (merge, no-filter floor), `1` (classic voting; main), `2` (newer default in `develop`, different rule: reprojection error 0.6 px + virtual-support prior; agent read only its header). Sweep views-to-fuse 2-4 and the depth/normal thresholds.
- No native external-depth input. Adapter: write per-view `depthNNNN.dmap` (header "D2", half-float depth, octahedral normals, uint8 confidence; **check field order against `ExportDepthDataRaw`**), a minimal Interface `.mvs` with poses, run `DensifyPointCloud --geometric-iters 0 --fusion-filter 1 --postprocess-dmaps 0 --view-neighbors-file ...`. **Smoke-test first (week 3-4)**: that `--geometric-iters 0` skips PatchMatch is an inference. Supply valid normals (compute from depth). Fallback: C++ shim calling `FuseDepthMaps`.
- Licence AGPL-3.0: release adapter + results, not OpenMVS code; check obligations if distributing modifications. Cite repo + version tag (no recommended citation); PatchMatch lineage (Shen 2013, Schonberger 2016, Barnes 2009, Galliani 2015, Xu & Tao 2019 — from memory, verify).
- Expected failure regimes (to test, not assume): extrinsic drift makes the reprojection test reject good points; bias consistent across views survives the vote.

**Family B — classical fusion** (from memory; verify every licence/reference)
| ID | Method | Notes |
|---|---|---|
| C1 | Weighted-average TSDF (Curless & Levoy 1996 / KinectFusion) via Open3D, known poses, no ICP | no-elimination floor |
| C2 | Per-pixel median across sensors (+ MAD outlier fraction) | robust-statistic baseline, 50% breakdown |
| C3 | COLMAP stereo-fusion geometric consistency (needs COLMAP depth/normal binaries + hand-built model) or own reimplementation of the same test | second voting baseline; compare against OpenMVS |
| C4 | Merrell 2007 visibility/free-space fusion | own reimplementation (~100-300 lines); closest classical analogue |
| C5 | TV-L1 range integration (Zach 2007/2008) | own reimplementation (primal-dual on voxel grid) |
| C6 | Probabilistic inlier-ratio fusion (Vogiatzis & Hernandez 2011 style / PSDF Fusion) | own reimplementation; no public PSDF code found (GitHub not searched directly); optional |
Cite only: ElasticFusion, BundleFusion, InfiniTAM, voxel hashing, Poisson. Exclude (need RGB): Goesele 2006, ACMH/ACMM/ACMP. Do not cite the "Ma/Zhang robust Kalman TSDF" — unverifiable. Optional speed/memory reference: VDBFusion / Voxblox / Nvblox. No open-source "sensor elimination before TSDF fusion" system was found (weak negative).

**Family C — learned fusion**
| ID | Method | Notes |
|---|---|---|
| L1 | **SenFuNet** (ECCV 2022; github.com/tfy14esa/SenFuNet; last push 2025-02) | only native multi-depth-sensor learned method with code. Learns per-voxel sensor weights; trained with GT SDF on Replica/CoRBS/Scene3D. Python 3.8, Open3D 0.13+, submodule, hardcoded paths. Report pretrained AND fine-tuned-on-clean-split. Pretrained-weight URL unverified. Licence BSD-style custom text. |
| L2 | RoutedFusion (CVPR 2020) **or** NeuralFusion (CVPR 2021), pick one | single-sensor learned fusion; ShapeNet-trained, synthetic zero-mean noise; no cross-sensor mechanism; run sensors sequentially. Last push 2021/2022. |
| L3 | optional: PromptDA (depth-prompted foundation model) | out-of-family proxy, needs RGB; label as such |
Skip: V-FUSE (needs MVS confidences/RGB; licence none; use only if you also run MVS), DI-Fusion (SLAM, no licence), Atlas/NeuralRecon/TransformerFusion (RGB only), NLSPN/CompletionFormer. No 2024-2026 learned multi-depth-sensor fusion with code was found (two searches; not proof).
Prediction to test (judgement, not measured): learned fusion trained on zero-mean noise and outliers will not cope with smooth, spatially consistent bias or extrinsic drift unless such faults are in its training distribution; add a "SenFuNet fine-tuned WITH fault-like bias" row as the strongest learned competitor.

**Family D — sensor-level elimination competitors (RAIM-style)**
B3 iterated leave-one-sensor-out normalised residual; B3b enumerated subset consensus (MSAC); B4 oracle; B5 estimate-and-correct (joint extrinsic/bias optimisation).

**Secondary (same-representation) — Gaussian splats:** 3DGS-MCMC with depth loss and same N, 2DGS if time. None runs depth-only as shipped (need RGB; placeholder-RGB + depth loss, plus an RGB-available row). SpotLessSplats/RobustSplat are photometric transient maskers: skip, justify in one sentence, B2 is the depth-space analogue. Research-only licences.

---

## 5. Hypotheses (pre-register at week 1)

- **H1** K≥3, one coherent fault ≥3σ: sensor-level elimination cuts Chamfer ≥25% vs the best of OpenMVS fusion (filter 1, tuned), C3/C4 voting, and B2, and lands within 20% of oracle — concentrated in low-overlap, faulty-dominant, and sub-noise regimes. With high overlap and large K, voting should tie (report the tie).
- **H2** Entropy criterion beats LOO χ² (B3) by ≥10 points of power at fixed 5% false-elimination rate, only where heteroscedastic/grazing/occlusion/non-Gaussian effects break χ²; ties under idealised Gaussian noise.
- **H3** Visibility + grazing weighting hold clean-run false elimination ≤5%; ablating either at least doubles it.
- **H5** (learned fusion) Pretrained SenFuNet and single-sensor learned fusion (RoutedFusion/NeuralFusion) degrade under smooth bias and extrinsic drift because they were trained on zero-mean noise/outliers; SenFuNet fine-tuned with fault-like bias narrows but does not close the gap in the low-overlap / faulty-dominant regimes. Falsified if fine-tuned SenFuNet matches elimination there — report it either way.
- **H4** Fixed-N MCMC resists absorbing the faulty sensor; adaptive densification at matched N loses ≥10 points of detection power; there is an N sweet spot.

Note on the premise: per-pixel voting does NOT fail simply because faults are coherent — two honest sensors still outvote it per pixel. It fails with ≤2 sensors covering a pixel, when the faulty sensor dominates coverage, when the fault is below the per-pixel noise floor (aggregation over ~10⁵ pixels detects 0.1–0.3σ), and when attribution is needed. Overlap and coverage share are controlled variables.

**Required honest-negative experiments:** majority faulty with same-sign bias; common-mode fault (shared extrinsic/clock); K=2 (attribution unidentifiable ≈50%); sub-noise faults; sparse independent outliers (voting should win); non-overlapping FOV.

---

## 6. Data and fault protocol

| Dataset | Role | Status |
|---|---|---|
| Skoltech3D (CVPR 2023) | **Primary real data.** Depth sensors: 2 phone ToF, Kinect v2 ToF, RealSense stereo; SL reference; per-sensor depth, poses, calibration. | CC BY-NC 4.0 (project page; README not confirmed). Tabletop objects, no room-scale multipath. Frames are independent viewpoints — define "time" by viewpoint order. Download size unknown; budget TB. |
| Replica | Synthetic multi-sensor rig (build per-sensor intrinsics, FOV, noise, extrinsics yourself); room-scale concave corners for multipath. | Research-only; **do not redistribute renderings**; release generator + seeds. |
| US3D / DFC2019 (Jacksonville, Omaha) | Optional satellite section. | **Licence conflict**: DataPort says CC BY 4.0, contest terms ban redistribution and require case-by-case publication approval (iadf_chairs@grss-ieee.org). ~289 GB. "UCSD" source not found — drop. |
| IARPA MVS3DM | Gong & Fritsch 2018 used it; it is **San Fernando, Argentina**, not Jacksonville/Omaha/UCSD. | Verify availability. |

Gong & Fritsch: "Point Cloud and Digital Surface Model Generation from High Resolution Multiple View Stereo Satellite Imagery", ISPRS Archives XLII-2, 363–370, 2018 (abstract only seen). Pair selection is one step, not its title.

Satellite framing: say "multi-source DSM fusion", not sensor fusion. Pairs sharing an image share its RPC bias → not independent; one bad image faults several pairs. Treat as secondary; first to cut.

**Fault grid (hash-derived seeds, ≥5 master seeds, JSON manifest published)**
- Rotation 0.5/1/2/5°, translation 1/2.5/5/10 cm; profile: step, ramp, or random walk.
- Constant bias 1–5% of median range; range-proportional β = 1/2/3/5% (both signs).
- Multipath surrogate: positive bias A·w(concavity), A = 2–10 cm tabletop, 5–30 cm rooms (heuristic, not physical).
- Faulted fraction: 1/3, 1/4, 2/4 (independent and same-direction), plus 0-fault control; sweep which sensor.
- Persistence: sensor-level persistent (onset 20%/40%) vs frame-level transient (10%/30% in bursts of 3–10); report separately.

**Metrics:** accuracy, completeness, Chamfer, F-score (Skoltech3D τ = 1/2/5 mm; Replica 2/5/10 cm); detection precision/recall, time-to-detect (median/90th, censored), false alarms per 100 clean frames, clean-run accuracy loss; define "elimination AUC" explicitly (non-standard). DSM: RMSE/MAE/completeness@1 m. Scene-level bootstrap CIs, paired Wilcoxon, dev/test scene split.

---

## 7. Schedule (18 weeks, ~10 h/wk)

| Wk | Work | Milestone |
|---|---|---|
| 1 | Read Sensors paper PDF; both-readings spec; derive entropy ↔ χ² relation; dataset licence checks; email IADF TC only if satellite stays; read YOGO + SenFuNet in full; pre-registration tag | M1 |
| 2 | Sensor simulators + fault injectors, deterministic, unit-tested | M2 |
| 3 | Multi-sensor depth-only fixed-N fit; alpha-mask loss; variance weighting. **OpenMVS adapter smoke test (dmap writer, `--geometric-iters 0`)** | M3 clean fit OK; OpenMVS fuses given depth |
| 4 | Baselines: OpenMVS (filters 0/1/2), C1 TSDF, C2 median, C3 COLMAP-style voting, C4 Merrell, B3/B3b/B4; harness; residual-autocorrelation figure | M4 |
| 5 | Elimination (both readings), visibility, grazing, null calibration, LOO warm start; first end-to-end | M5 |
| 6 | **Gate 1** on 3 dev scenes × 3 fault types | go / pivot |
| 7 | Full synthetic sweeps | H1 curves |
| 8 | Ablations (fixed-N vs ADC, N sweep, visibility/grazing) | H3, H4 |
| 9 | Breakdown experiments; **Gate 2** | honest negatives |
| 10 | Skoltech3D real data with injected faults; **Gate 3** | real table |
| 11 | **SenFuNet** (pretrained + fine-tuned clean + fine-tuned with bias), C5 TV-L1, B5; RoutedFusion/NeuralFusion; 3DGS-MCMC/2DGS secondary | learned-family table |
| 12 | Satellite DSM section (only if licence approval in hand) else dynamics test | optional |
| 13 | **Experiment freeze** | tag |
| 14–16 | Draft, figures, related work, limitations, reproducibility package, AI-use disclosure | submission-ready |
| 17–18 | Buffer | |

**Gate 1 (wk 6):** Go if elimination beats B0, B1/B2 on coherent faults AND entropy beats LOO χ² in ≥1 regime by the H2 margin.
- Pivot P1: entropy ≈ χ² → the paper is "sensor-level exclusion in splat space with visibility/grazing weighting"; entropy becomes an equivalence result.
- Pivot P2: no win over voting in the H1 regimes → benchmark-and-analysis paper ("when does per-pixel voting fail").

**Cut order:** satellite → dynamics → second real dataset → 3DGS-MCMC/2DGS secondary → C6 / TV-L1 → stereo sensor → B5 (discussion only). OpenMVS, one classical consistency baseline (C3/C4), and SenFuNet are core and never cut.
**Never cut:** breakdown experiments, clean-run false-elimination, LOO χ² baseline, the Eq. 14/Alg. 1 fix.

---

## 8. Paper outline (~9 pp + refs; IEEE Access recommends < 20)
Abstract · 1 Intro · 2 Related work (multi-sensor splatting, robust splatting, depth fusion/voting, RAIM/Byzantine) · 3 Problem and fault model · 4 Method (shared surface, residuals, corrected criterion, elimination, persistence) · 5 Setup · 6 Results (H1–H4, breakdown, real data) · 7 Limitations and failure conditions · 8 Conclusion · Appendix (derivations, Eq. 14 fix, hyperparameters).
Figures: F1 vote-vs-sensor-level error maps; F2 pipeline; F3 residual autocorrelation coherent vs sparse; F4 error vs magnitude; F5 power at fixed false-elimination, entropy vs χ²; F6 breakdown phase diagram; F7 K=2 non-identifiability; F8 real qualitative; F9 ablations.

---

## 9. IEEE Access facts (agent: venue)
Verified on official pages: gold OA; APC $2,160 (+ taxes; 5%/20% member discounts only); no page limit, <20 pp recommended; single-anonymised, ≥2 reviews, ~4–6 weeks submission-to-publication, acceptance ~20%; reviewers judge technical soundness, not necessarily high novelty; no formal data/code statement required; preprints allowed (arXiv named in IEEE-wide policy, not on Access pages); AI-assisted writing must be disclosed in Acknowledgments; fabricated references → rejection; public ORCID required.
Not verified / yours to resolve: Virginia Tech–IEEE read-and-publish agreement (ask VT libraries; VT OASF announced closed Jul 2025); current IEEE copyright-form (eCF) wording and any publication-approval obligations that apply to you — check the live form. No Access conflict-of-interest page found. Not legal advice. Comparable-paper titles came from catalogue metadata only.

---

## 10. Open items checklist
- [ ] Re-read Sensors 23(23):9309 PDF equations to confirm ≤/≥ in Alg. 1 and the entropy normalisation.
- [ ] Read YOGO, SenFuNet, arXiv 2306.11048 in full; IEEE Xplore/Scholar novelty pass.
- [ ] Verify every citation (agent-from-memory refs especially) before use.
- [ ] Skoltech3D licence text in the repo; Replica redistribution (generator-only release).
- [ ] Decide whether satellite stays in scope given licence approval and analogy weakness.
- [ ] VT APC route; read live IEEE copyright form; check employment IP/publication-review terms.
- [ ] Smoke-test OpenMVS external-depth path; check `.dmap` field order in `ExportDepthDataRaw`; check AGPL obligations for the adapter.
- [ ] SenFuNet: locate pretrained-weight URL, read full licence, confirm it runs on current CUDA/PyTorch; plan GT-SDF generation for fault-fine-tuning.
- [ ] Verify all classical/learned licences and references (mostly from memory).
- [ ] Reproducibility package: seeds, manifests, configs, tuning budgets, GPU-hours.
