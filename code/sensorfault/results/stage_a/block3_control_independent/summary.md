# Stage A summary: matrixcity block3, source exact_independent, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 1 [0.917, 1.07], pass.
- Noise model (class): white pixel noise sigma (global 0.03956) plus a record floor tau (global 0.006098, per-frame median 0.006098, p10-p90 0.00207-0.01149); floor shape log g: roof +1.547, steep -3.238, log_pixels -0.065, log_incidence -0.949, log_depth +7.784 (g p10-p90 0.08-5.23).
- A8 s_z by pixels per record (held-out land): 4-5 px 0.954, 6-8 px 0.976, 9-184 px 1.09; by class: roof 0.898, ground 1.02.

## Alignment (A5)

Mode metric, gauge k = 0.9754, beta = 0, last change 0.000435 (converged).
Block datum: calibration (factor 1.0257).
Median relative depth residual: label-free with parallax datum 0.0341, after datum 0.0301, oracle per-frame fit 0.03.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 7.26 m; additivity max relative difference 5.86e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 1.08 [1.02, 1.15] | 1.12 | 0.611 | 0.893 | 0.0513 | 3.08 | 0.0402 |
| land | 628 | 1.13 [1.03, 1.23] | 1.18 | 0.581 | 0.854 | 0.0794 | 2.43 | 0.0956 |
| water | 501 | 1.04 [0.979, 1.09] | 1.03 | 0.649 | 0.942 | 0.0175 | 3.9 | 0.00261 |
| roof | 162 | 1.13 [0.97, 1.28] | 1.11 | 0.574 | 0.883 | 0.0734 | 3.99 | 0.0345 |
| ground | 466 | 1.13 [1.06, 1.23] | 1.21 | 0.584 | 0.843 | 0.0814 | 1.88 | 0.122 |
| contrast_q1 | 134 | 1.13 [1.07, 1.2] | 1.14 | 0.597 | 0.858 | 0.0833 | 4.11 | 0.056 |
| contrast_q2 | 139 | 0.991 [0.926, 1.06] | 1.03 | 0.676 | 0.906 | 0.0215 | 3.03 | 0.00274 |
| contrast_q3 | 163 | 1.16 [1.07, 1.28] | 1.17 | 0.564 | 0.877 | 0.0948 | 1.9 | 0.0818 |
| contrast_q4 | 107 | 1.26 [1, 1.61] | 1.38 | 0.477 | 0.729 | 0.15 | 1.83 | 0.381 |
| contrast_q5 | 85 | 1.18 [1.08, 1.34] | 1.25 | 0.565 | 0.871 | 0.0889 | 0.562 | 0.173 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 35 | 1.01 | 1.04 [0.975, 1.13] | 0.902 |
| 3 | 33 | 1.02 | 1.04 [0.991, 1.13] | 0.895 |
| 5 | 34 | 1.02 | 1.04 [0.983, 1.14] | 0.9 |
| 9 | 34 | 1.02 | 1.04 [0.983, 1.14] | 0.9 |

## Repetition (A10)

r(s), same strip: s=1: 0.0225, s=2: 0.00269, s=3: 0.00159, s=4: -0.00272, s=5: -0.00916, s=6: 0.0148, s=7: -0.00587, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00251; cross heading r = -0.00138.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.0103 [-0.00172, 0.0319] | 0.00519 [-0.00485, 0.011] |
| 1 | 0.00114 [-0.00317, 0.0111] | 0.00791 [-0.00535, 0.017] |
| 2 | -0.000389 [-0.00467, 0.00976] | 0.00882 [-0.00702, 0.0191] |
| 3 | -0.00318 [-0.0227, 0.00473] | 0.012 [-0.0075, 0.0272] |
| 5 | 0.0136 [-0.00697, 0.097] | -0.00555 [-0.0329, 0.0388] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = -0.00318 [-0.0227, 0.00473]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.494 | 0.5 | 0.5 | 0.987 [0.792, 1.05] | 0.0673 |
| 3-4 | 228 | 0.318 | 0.284 | 0.284 | 1.12 [0.957, 1.31] | 0.105 |
| 5-8 | 230 | 0.12 | 0.155 | 0.155 | 0.774 [0.601, 1.35] | 0.211 |
| 9-16 | 275 | 0.0873 | 0.0823 | 0.0823 | 1.06 [0.977, 1.28] | 0.0751 |
| 17+ | 624 | 0.0545 | 0.0412 | 0.0412 | 1.32 [0.95, 1.6] | 0.358 |

## Residual parallax on textured land (A15)

p90_px_scale1: 22.1, p90_px_scale2: 11, p90_px_scale4: 5.52, p90_px_scale8: 2.76, n_pixels: 3.5e+05

Runtime 6.6 min.
