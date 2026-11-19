# Stage A summary: matrixcity block3, source exact_repeated, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.19 [0.925, 1.57], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 0.9923, beta = 0, last change 0.224 (not below 0.1%).
Block datum: calibration (factor 1.0077).
Median relative depth residual: label-free with parallax datum 0.0101, after datum 0.00743, oracle per-frame fit 0.0074.

## Records (A7)

100682 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.33 m; additivity max relative difference 7.08e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 2.48 [2.18, 2.81] | 46.9 | 0.148 | 0.323 | 0.381 | 3.3e+03 | 3.28e+03 |
| land | 628 | 2.74 [2.24, 3.51] | 62.8 | 0.132 | 0.283 | 0.394 | 5.92e+03 | 5.91e+03 |
| water | 501 | 2.24 [1.87, 2.57] | 2.45 | 0.168 | 0.373 | 0.364 | 7.24 | 4.79 |
| roof | 162 | 2.49 [1.91, 3.37] | 3.52 | 0.148 | 0.309 | 0.377 | 16.7 | 13.3 |
| ground | 466 | 2.85 [2.34, 3.71] | 72.9 | 0.127 | 0.275 | 0.4 | 7.97e+03 | 7.96e+03 |
| contrast_q1 | 134 | 2.47 [2.1, 2.7] | 3.06 | 0.179 | 0.336 | 0.368 | 12.3 | 9.18 |
| contrast_q2 | 139 | 2.49 [2.07, 3.02] | 13.7 | 0.137 | 0.302 | 0.376 | 281 | 274 |
| contrast_q3 | 163 | 2.69 [1.96, 4.25] | 23.9 | 0.141 | 0.313 | 0.388 | 854 | 847 |
| contrast_q4 | 107 | 3.52 [2.59, 5.35] | 96.9 | 0.0935 | 0.196 | 0.432 | 1.41e+04 | 1.41e+04 |
| contrast_q5 | 85 | 3.81 [2.38, 6.16] | 126 | 0.0824 | 0.224 | 0.428 | 2.39e+04 | 2.39e+04 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 82 | 68.9 | 2.19 [1.88, 2.66] | 0.392 |
| 3 | 80 | 68.6 | 2.24 [1.94, 2.66] | 0.375 |
| 5 | 80 | 68.5 | 2.24 [1.94, 2.66] | 0.375 |
| 9 | 80 | 68.5 | 2.24 [1.94, 2.66] | 0.375 |

## Repetition (A10)

r(s), same strip: s=1: 0.231, s=2: 0.238, s=3: 0.233, s=4: 0.232, s=5: 0.225, s=6: 0.203, s=7: 0.256, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.226; cross heading r = 0.222.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.231 [0.186, 0.299] | -0.0186 [-0.0643, 0.000657] |
| 1 | 0.231 [0.19, 0.311] | -0.0237 [-0.078, 0.00538] |
| 2 | 0.23 [0.184, 0.318] | -0.0289 [-0.0983, 0.00399] |
| 3 | 0.229 [0.183, 0.344] | -0.0357 [-0.119, 0.0123] |
| 5 | 0.229 [0.151, 0.427] | -0.0726 [-0.232, 0.00493] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.229 [0.183, 0.344]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.672 | 0.614 | 0.5 | 1.09 [0.902, 1.23] | 1.23 |
| 3-4 | 228 | 0.403 | 0.448 | 0.284 | 0.899 [0.714, 1.1] | 1.1 |
| 5-8 | 230 | 0.361 | 0.348 | 0.155 | 1.04 [0.792, 1.24] | 1.13 |
| 9-16 | 275 | 0.227 | 0.292 | 0.0823 | 0.776 [0.535, 1.24] | 0.732 |
| 17+ | 624 | 0.29 | 0.261 | 0.0412 | 1.11 [0.815, 1.37] | 1.2 |

## Residual parallax on textured land (A15)

p90_px_scale1: 39.7, p90_px_scale2: 19.9, p90_px_scale4: 9.93, p90_px_scale8: 4.96, n_pixels: 3.5e+05

Runtime 9.4 min.
