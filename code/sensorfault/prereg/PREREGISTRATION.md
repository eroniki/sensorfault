# Pre-registration (DRAFT, status: `preregistered_draft`)

Project: auditing monocular depth with per-source information on Gaussian surfels (design v4.0, see `PLAN.md`).
This file is frozen by `git tag prereg-v4` BEFORE any held-out tile or test scene is scored (see `TAGGING.md`).
Any later change goes into `## Deviations` with date and reason; the tagged text is never rewritten.
No earlier pre-registration was tagged; the drafts for designs v1 and v3 are in the git history.

Public datasets, public pretrained models and public code only. No fault labels are used for fitting at any stage;
exact or laser depth is used for evaluation only.

## 1. Hypotheses, predictions, falsification, thresholds

The full protocol is `docs/learned-indicator/experiment-design.md` (decision rules §1, data and splits §2,
hypotheses §3, metrics §4, faults §5, baselines §6). Machine-readable copies of the hypotheses of the first paper are
in `configs/stage*.yaml` and `configs/real_scenes_check.yaml` (`hypotheses[].supported_if`, `refuted_if`, `stop`).
Where a config and the protocol differ, the protocol wins and the config is corrected under Deviations.

| ID | Stage | Prediction | Primary metric | Supported if | Refuted if (stop) |
|---|---|---|---|---|---|
| H1 | A | A depth-only information map is overconfident | robust scale s_z of the standardised placement error, entities with n >= 5 frames | lower bound > 1.25 | upper bound <= 1.25 after the consensus gate (S1) |
| H2 | A | Fused variance is rho + (1 - rho)/n times the single-frame variance | realised / predicted per n-bin | interval in [0.67, 1.5] | rho upper bound < 0.1, or ratio outside [0.5, 2] (S1) |
| H3 | 0, B | The image record recovers an injected placement | relative error of the recovered offset | <= 0.25 up to the 90th percentile of residual parallax on textured land | no scale recovers it (S2) |
| H4 | B | Realistic faults carry energy | noncentrality at the 75th-percentile surface term | >= 11 | < 5 (S2) |
| H5 | B | An appearance regime keeps a fault and renders healthy surface | absorbed fraction; PSNR loss | <= 0.5, retained >= 0.5, loss <= 1 dB | no such regime (S3) |
| H6 | B | Stated image information is honest on blank surface | nominal / realised information | in [1/3, 3] | reported, no stop |
| H7 | C | The fused map is calibrated | s_z, coverage, ECE, surface log-score | s_z in [0.8, 1.25]; coverage within 0.08 / 0.04; ECE <= 0.05 | s_z > 1.5 or log-score no better than depth only (S4) |
| H8 | C | Gating adds to saturated fusion (secondary) | share of fault-affected entities with z2 > 11.345, gated / ungated | <= 0.5, upper bound < 0.8 | upper bound >= 1, or a decaying weight matches it (S5) |
| H9 | C | Repeated faults are visible to the tension, not to consensus | AUROC | >= 0.9 against <= 0.6 | consensus AUROC >= 0.85 (S5) |
| H10 | C | The energy has its stated null | mean T/3; exceedance of 7.815 | [0.85, 1.15]; [0.035, 0.075] | outside both (S4) |
| H14 | 0 | Records are additive | relative Frobenius difference | <= 1e-6 (float64) | arithmetic error |

Decision rule: a hypothesis is supported or refuted only when the 95% bootstrap interval of its primary metric lies
wholly on one side of the threshold; a stop rule fires only on refuted; an inconclusive result allows one enlargement
of the held-out data and is then final. All outcomes are reported, including refuted ones and the fallback paper they
lead to (`docs/learned-indicator/paper-plan.md` §4).

## 2. Primary analysis, metrics, statistics

* Unit of analysis: the 64 m ground tile (MatrixCity) or the scene (ETH3D). Tile bootstrap, 10 000 resamples, 95%;
  with several blocks, blocks are resampled first, then tiles. Paired comparisons between maps on the same tiles.
* Surface log-score: negative log density of the true placement of each entity on the signed normal error, averaged
  over entities (nats per entity). Methods without a stated variance get the best constant variance fitted on
  calibration tiles.
* Multiplicity: each hypothesis is one pre-declared family with one primary metric; per-class and per-model results
  are secondary and labelled so. Benjamini-Hochberg within the fault grid.
* Seeds: master seeds `[101..105]`; fault manifests are content-hashed and released.

## 3. Tuning protocol (dev data only)

1. Thresholds, the block size B, the appearance regime, tile sizes and every baseline's settings are fixed on dev data
   (MatrixCity block 3 calibration tiles; ETH3D courtyard, facade, office, relief) with the same tuning budget per
   method; budgets are reported.
2. Identical inputs for all methods: same frames, poses, depth maps, fault realisations and tiles.
3. Reimplemented baselines are labelled as such and released.
4. After `prereg-v4` no change to thresholds, metrics, strata or the baseline list without a dated entry under
   Deviations.

## 4. Dev / test split

* MatrixCity: block 3 is dev. Test is the three lowest-indexed other aerial blocks with depth and RGB, chosen before
  any is opened. If none is obtained, block 3 is split by a checkerboard of 64 m tiles into calibration and held-out
  tiles with a one-tile buffer, and every table is marked "same scene".
* ETH3D: dev scenes as above; the other nine are test.
* UseGeo (optional): set 1 dev, sets 2 and 3 test.

## 5. Null calibration

The energy's null is checked, not assumed (H10): on healthy held-out entities, and on a placebo in which the depth
frames are split into two halves and tested against each other. Detection thresholds are set at a 5% false-alarm
rate on calibration tiles and reported with the realised rate on held-out tiles, never retuned.

## 6. Exclusions and stopping rules

* A run is excluded only for non-numeric failure (crash, out of memory) and is reported as a failure.
* No optional stopping on held-out results. The cut order of `PLAN.md` §6 applies to scope and is announced under
  Deviations.

## 7. Deviations (append only)

| Date | Change | Reason |
|---|---|---|
| 2026-11-19 | H2 is tested on the equal-weight mean of the frames' placement errors; the information-weighted map's ratios are reported alongside. | The law rho + (1 - rho)/n is the variance of the equal-weight mean, and r(s) is pooled the same way. The information-weighted mean falls below 1/n even for independent errors when record variances differ, so it does not test the law. |
| 2026-11-19 | rho_hat(B) is the pair-count weighted mean of r(s) over s > B (was the unweighted mean). | Separations near the strip ends have few pairs; unweighted they carry as much weight as separations with hundreds. |
| 2026-11-19 | Single-frame noise model (A8): white pixel noise sigma_i from the within-record residual plus a record-level floor tau_i h, Sigma = sigma_i^2 Lambda0^-1 + (tau_i h)^2 I, both calibrated per frame on calibration tiles (was white pixel noise only). | Monocular depth errors are smooth, so pixel errors within one record are nearly fully correlated; with white noise only the stated information grows with the pixel count and the single-frame s_z with it. |
| 2026-11-19 | The record floor's size depends on record covariates: log g = beta . (roof, steep, log pixels, log incidence, log depth), fitted on calibration land by a Student-t (nu = 4) likelihood; Sigma = sigma_i^2 Lambda0^-1 + (tau_i g h)^2 I, with tau_i set per frame by the median rule (`--noise class`, the default). | With one floor per frame the single-frame s_z still differed between surface classes and grew with pixels per record on held-out land, so the shared error depends on the record, not only on the frame. |
