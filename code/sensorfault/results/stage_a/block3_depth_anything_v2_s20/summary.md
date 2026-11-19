# Stage A summary: matrixcity block3, source cache (depth_anything_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.26 [1.03, 1.51], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.0009628) plus a record floor tau (global 0.2594, per-frame median 0.2594, p10-p90 0.1275-0.4006).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.23, 6-8 px 1.18, 9-184 px 1.49.

## Alignment (A5)

Mode inverse, gauge k = 1.136e+04, beta = 0, last change 0.4 (not below 0.1%).
Block datum: calibration (factor 3.9184).
Median relative depth residual: label-free with parallax datum 0.776, after datum 0.366, oracle per-frame fit 0.329.

## Records (A7)

100683 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.19 m; additivity max relative difference 7.99e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 4.04 [2.98, 5.7] | 6.74 | 0.167 | 0.234 | 0.366 | 75.7 | 60.9 |
| land | 628 | 4.36 [3.25, 5.92] | 7.97 | 0.153 | 0.212 | 0.377 | 102 | 87.6 |
| water | 501 | 3.81 [2.17, 7.03] | 4.76 | 0.184 | 0.261 | 0.351 | 43.1 | 27.9 |
| roof | 162 | 5.21 [1.89, 7.6] | 9.73 | 0.16 | 0.198 | 0.373 | 148 | 134 |
| ground | 466 | 4.12 [2.98, 5.81] | 7.26 | 0.15 | 0.217 | 0.379 | 85.4 | 71.6 |
| contrast_q1 | 134 | 3.31 [2.4, 4.61] | 7.95 | 0.224 | 0.261 | 0.322 | 103 | 87.1 |
| contrast_q2 | 139 | 5.22 [4.1, 6.24] | 8.86 | 0.129 | 0.158 | 0.392 | 125 | 110 |
| contrast_q3 | 163 | 5.24 [3.3, 7.18] | 8.07 | 0.129 | 0.178 | 0.395 | 104 | 89.9 |
| contrast_q4 | 107 | 3.82 [2.73, 5.45] | 7.03 | 0.131 | 0.243 | 0.392 | 80 | 66.7 |
| contrast_q5 | 85 | 4.43 [2.69, 5.5] | 7.37 | 0.153 | 0.247 | 0.385 | 86.3 | 74 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 86 | 1.08 | 3.62 [2.6, 4.85] | 0.283 |
| 3 | 87 | 1.13 | 3.55 [2.6, 4.71] | 0.286 |
| 5 | 88 | 1.09 | 3.28 [2.43, 4.99] | 0.296 |
| 9 | 87 | 1.08 | 3.52 [2.58, 4.99] | 0.285 |

## Repetition (A10)

r(s), same strip: s=1: 0.916, s=2: 0.924, s=3: 0.959, s=4: 0.979, s=5: 1.15, s=6: 1.22, s=7: 1.28, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.44; cross heading r = 0.0262.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.94 [0.76, 1.12] | 0.44 [0.347, 0.509] |
| 1 | 0.959 [0.748, 1.2] | 0.438 [0.347, 0.505] |
| 2 | 0.993 [0.727, 1.35] | 0.425 [0.31, 0.489] |
| 3 | 1.04 [0.613, 1.64] | 0.41 [0.204, 0.5] |
| 5 | 1.23 [0.202, 2.72] | 0.645 [-0.0014, 0.946] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 1.04 [0.613, 1.64]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.99 | 1 | 0.5 | 0.99 [0.948, 1.2] | 0.976 |
| 3-4 | 228 | 0.915 | 1 | 0.284 | 0.915 [0.799, 1.25] | 0.868 |
| 5-8 | 230 | 0.784 | 1 | 0.155 | 0.784 [0.42, 1.22] | 0.786 |
| 9-16 | 275 | 0.779 | 1 | 0.0823 | 0.779 [0.493, 1.16] | 0.667 |
| 17+ | 624 | 0.503 | 1 | 0.0412 | 0.503 [0.341, 0.823] | 0.194 |

## Residual parallax on textured land (A15)

p90_px_scale1: 467, p90_px_scale2: 233, p90_px_scale4: 117, p90_px_scale8: 58.3, n_pixels: 3.5e+05

Runtime 6.8 min.
