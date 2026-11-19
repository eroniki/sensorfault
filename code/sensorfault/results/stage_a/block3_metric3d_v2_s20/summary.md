# Stage A summary: matrixcity block3, source cache (metric3d_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.2 [0.966, 1.43], FAIL (noise model at fault, H1 undefined).
- Noise model (class): white pixel noise sigma (global 0.007297) plus a record floor tau (global 0.04557, per-frame median 0.04557, p10-p90 0.03359-0.07399); floor shape log g: roof +0.853, steep +1.050, log_pixels +0.171, log_incidence -0.097, log_depth +0.951 (g p10-p90 0.568-2.35).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.25, 6-8 px 1.2, 9-184 px 1.14; by class: roof 1.11, ground 1.22.

## Alignment (A5)

Mode metric, gauge k = 6.61, beta = 0, last change 0.000515 (converged).
Block datum: calibration (factor 1.0814).
Median relative depth residual: label-free with parallax datum 0.159, after datum 0.157, oracle per-frame fit 0.113.

## Records (A7)

100690 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.931 m; additivity max relative difference 5.37e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 10.4 [4.43, 28.5] | 24.8 | 0.0797 | 0.135 | 0.441 | 928 | 915 |
| land | 628 | 3.32 [2.68, 4.67] | 7.44 | 0.143 | 0.242 | 0.394 | 86.1 | 75.5 |
| water | 501 | 36.5 [27.5, 49] | 36.4 | 0 | 0 | 0.5 | 1.98e+03 | 1.97e+03 |
| roof | 162 | 3.11 [1.82, 5.32] | 4.49 | 0.179 | 0.302 | 0.374 | 35.5 | 24.2 |
| ground | 466 | 3.39 [2.73, 5.2] | 8.22 | 0.131 | 0.221 | 0.401 | 104 | 93.6 |
| contrast_q1 | 134 | 3.68 [2.28, 6.53] | 5.89 | 0.119 | 0.261 | 0.405 | 55.7 | 45.3 |
| contrast_q2 | 139 | 3.39 [2.39, 4.99] | 5.42 | 0.144 | 0.245 | 0.403 | 47.5 | 37.5 |
| contrast_q3 | 163 | 3.58 [3.03, 5.02] | 5.58 | 0.123 | 0.166 | 0.415 | 49.5 | 40.1 |
| contrast_q4 | 107 | 2.82 [2.54, 4.41] | 4.7 | 0.15 | 0.252 | 0.383 | 36.6 | 27.1 |
| contrast_q5 | 85 | 2.63 [2.11, 3.87] | 14.8 | 0.212 | 0.341 | 0.336 | 330 | 318 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 81 | 1.14 | 3.09 [2.39, 3.85] | 0.279 |
| 3 | 82 | 1.15 | 3.04 [2.29, 3.94] | 0.294 |
| 5 | 80 | 1.14 | 3.07 [2.29, 3.99] | 0.281 |
| 9 | 80 | 1.14 | 3.07 [2.29, 3.99] | 0.281 |

## Repetition (A10)

r(s), same strip: s=1: 0.889, s=2: 0.825, s=3: 0.702, s=4: 0.498, s=5: 0.265, s=6: 0.152, s=7: 0.0333, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.802; cross heading r = 0.712.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.783 [0.675, 0.896] | 0.495 [0.392, 0.566] |
| 1 | 0.703 [0.579, 0.834] | 0.44 [0.356, 0.504] |
| 2 | 0.582 [0.452, 0.77] | 0.357 [0.287, 0.412] |
| 3 | 0.412 [0.282, 0.621] | 0.257 [0.178, 0.327] |
| 5 | 0.145 [0.0658, 0.376] | 0.238 [-0.238, 0.388] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.412 [0.282, 0.621]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.926 | 0.706 | 0.5 | 1.31 [1.13, 1.45] | 1.25 |
| 3-4 | 228 | 0.773 | 0.579 | 0.284 | 1.33 [0.825, 1.8] | 1.35 |
| 5-8 | 230 | 0.863 | 0.503 | 0.155 | 1.72 [1.26, 2.24] | 1.72 |
| 9-16 | 275 | 0.772 | 0.461 | 0.0823 | 1.68 [1.04, 2.43] | 1.59 |
| 17+ | 624 | 0.893 | 0.436 | 0.0412 | 2.05 [1.36, 2.95] | 2.2 |

## Residual parallax on textured land (A15)

p90_px_scale1: 110, p90_px_scale2: 55, p90_px_scale4: 27.5, p90_px_scale8: 13.8, n_pixels: 3.5e+05

Runtime 7.1 min.
