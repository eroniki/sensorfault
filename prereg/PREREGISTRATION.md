# Pre-registration (DRAFT, status: `preregistered_draft`)

Project: sensor-level fault elimination in a shared Gaussian-splat surface (see `PLAN.md`).
This file is frozen by `git tag prereg-v1` BEFORE any test-split scene is run (see `TAGGING.md`).
Any later change goes into `## Deviations` with date and reason; the tagged text is never rewritten.

Public datasets, public sensor models and public code only.

## 1. Hypotheses, predictions, falsification, thresholds

Exact machine-readable copies live in `configs/h*_*.yaml` (`hypothesis.prediction`, `hypothesis.falsification_criterion`).

| ID | Prediction (PLAN.md section 5) | Primary metric | Falsified if |
|---|---|---|---|
| H1 | K>=3, one coherent fault >=3 sigma: elimination cuts Chamfer >=25% vs best of {OpenMVS filter 1 tuned, C3/C4 voting, B2 median}, within 20% of oracle; concentrated in low-overlap, faulty-dominant, sub-noise regimes; ties at high overlap / large K | Chamfer, relative reduction | CI lower bound of reduction <25% in the pre-declared regimes, or median >20% above oracle. High-overlap tie is a predicted null |
| H2 | Admission/entropy criterion beats LOO chi^2 (B3) by >=10 power points at FER=5%, only in heteroscedastic/grazing/occlusion/non-Gaussian regimes; ties for idealised Gaussian | Power at FER 5%; `elimination_curve_auc` | Paired power difference CI lower bound <10 points in every non-idealised regime (then pivot P1) |
| H3 | Visibility + grazing weighting keep clean-run false elimination <=5%; ablating either >= doubles it | Clean-run false-elimination rate | 95% upper bound >5% with both on, or ablation ratio <2 |
| H4 | Fixed-N MCMC resists absorbing the faulty sensor; ADC at matched N loses >=10 power points; N sweet spot exists | Power at FER 5% | Difference CI lower bound <10 points, or power monotone in N |
| H5 | Pretrained SenFuNet / RoutedFusion (or NeuralFusion) degrade under smooth bias & extrinsic drift; SenFuNet fine-tuned with bias narrows but does not close the gap in low-overlap / faulty-dominant regimes | Chamfer | Fine-tuned-with-bias SenFuNet within 10% of elimination (Wilcoxon not rejecting) in those regimes |

Required honest-negative experiments (`configs/breakdown_*.yaml`): majority same-sign faulty, common-mode, K=2,
sub-noise, sparse-outlier control (voting should win), non-overlapping FOV. All reported regardless of outcome.

Gates (PLAN.md section 7): Gate 1 week 6 (go / pivot P1 entropy~chi^2 / pivot P2 no win over voting), Gate 2 week 9, Gate 3 week 10.

## 2. Primary analysis, metrics, statistics

* Chamfer (symmetric mean distance), accuracy, completeness, F-score at tau (rooms 2/5/10 cm; Skoltech3D 1/2/5 mm).
* Detection precision/recall per sensor; time-to-detect (median/90th, Kaplan-Meier, censored at horizon);
  false alarms per 100 clean frames; clean-run accuracy loss; latency per method.
* `elimination_curve_auc`: normalised partial AUC of power vs clean-run false-elimination rate over FER in [0, 0.2]
  (definition in `sensorfault.eval.metrics`).
* Unit of analysis = SCENE. Seeds are averaged within a scene, then percentile bootstrap (10 000 resamples, 95%) over scenes;
  paired Wilcoxon signed-rank for method comparisons. Equivalence claims use the margins stated per hypothesis.
* Completeness caveat: ground truth is the union of per-sensor clean visible surfaces; excluding a sensor can lower
  completeness. Report accuracy and completeness separately next to Chamfer.
* >=5 master seeds `[101..105]`; fault seeds `SeedSequence([master, stable_hash(sensor_id), fault_id])`; manifests (JSON, content-hashed) are released.
* Multiplicity: H1-H5 are separate pre-declared families; within a hypothesis, regimes are pre-declared (above); no other
  subgroup is claimed confirmatory. Everything else is labelled exploratory.

## 3. Tuning protocol (dev scenes only)

1. All thresholds, hyper-parameters, and baseline sweeps (OpenMVS views-to-fuse 2-4, depth/normal thresholds, filter 0/1/2;
   TSDF voxel/trunc; voting thresholds) are tuned on DEV scenes only, with the same tuning budget (number of configurations
   and GPU-hours) for every method; budgets are reported.
2. Identical inputs for all methods: same fault-injected depth maps, same (possibly drifted) poses and intrinsics, same view graph,
   same per-pixel confidence (or none), same fault realisations.
3. Reimplementations are labelled and released. Offline methods are accuracy baselines; latency reported for all, with the
   < 100 ms/frame target misses stated.
4. After `prereg-v1` no change to thresholds, metrics, regimes, or baseline list without a dated entry under Deviations.

## 4. Dev / test split

* Synthetic: dev scenes `1001-1003`, test scenes `2001-2006` (`configs/base.yaml`); scene seed fully determines geometry/rig.
* Skoltech3D / Replica: sorted scene names; `hash(name) % 4 == 0` -> dev (<=25%), the rest test; list written to
  `configs/datasets/*.yaml` before tagging.
* The runner refuses `--split test` unless tag `prereg-v1` exists.

## 5. Null calibration (5% false elimination)

For each method with a threshold on a per-sensor statistic:
1. Collect the statistic on CLEAN (no fault) dev runs, per sensor, across scenes and seeds (>= 200 sensor-runs per regime).
2. Permutation null: randomly permute sensor labels / frame order within clean runs (`n_permutations: 200`), recompute the
   max-over-sensors statistic per run, and set the threshold at the 95th percentile of that null distribution, so that the
   per-run false-elimination probability is 5%.
3. Fix the threshold; verify on held-out clean DEV runs (rate with Clopper-Pearson CI); freeze it into the tag.
4. On test scenes report the realised clean-run false-elimination rate with CI, never re-tuning.
The same procedure is applied to LOO chi^2 (B3/B3b) so H2 compares methods at equal FER.

## 6. Variants and the Eq. 14 / Alg. 1 fix

The corrected admission criterion (Lambda_k(C)=|S| Z_{C u k}/Z_C >= 1, seed by max Z_ij, largest-Lambda expansion, min-entropy
selection with <=F removals; `sensorfault.elim`) is the PRE-REGISTERED method. The literal Alg. 1 reading is run only as ablation A8.
Verify against the Sensors 23(23):9309 PDF before submission (PLAN.md open items).

## 7. Exclusions and stopping rules

* A run is excluded only for non-numeric failure (crash, OOM) and is reported as a failure, not dropped silently; fuser skips
  (missing dependency) are logged per run.
* No optional stopping on test results. Cut order (PLAN.md section 7) applies to scope, announced under Deviations.

## 8. Deviations (append only)

| Date | Change | Reason |
|---|---|---|
