# Stage A summary: matrixcity block3, source cache (depth_anything_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.18 [0.907, 1.45], FAIL (noise model at fault, H1 undefined).
- Noise model (class): white pixel noise sigma (global 0.0009628) plus a record floor tau (global 0.2671, per-frame median 0.2671, p10-p90 0.1323-0.429); floor shape log g: roof +0.493, steep +0.006, log_pixels -0.120, log_incidence -0.096, log_depth -0.105 (g p10-p90 0.869-1.43).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.1, 6-8 px 1.1, 9-184 px 1.4; by class: roof 0.879, ground 1.24.

## Alignment (A5)

Mode inverse, gauge k = 1.136e+04, beta = 0, last change 0.4 (not below 0.1%).
Block datum: calibration (factor 3.9184).
Median relative depth residual: label-free with parallax datum 0.776, after datum 0.366, oracle per-frame fit 0.329.

## Records (A7)

100683 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.19 m; additivity max relative difference 5.83e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 3.82 [2.76, 5.32] | 5.55 | 0.177 | 0.252 | 0.355 | 54.1 | 39.6 |
| land | 628 | 3.65 [2.68, 5.25] | 5.82 | 0.175 | 0.248 | 0.354 | 57.8 | 44 |
| water | 501 | 3.91 [2.24, 7.67] | 5.2 | 0.18 | 0.257 | 0.355 | 49.5 | 34.1 |
| roof | 162 | 2.93 [1.07, 4.45] | 5.4 | 0.204 | 0.278 | 0.321 | 51.9 | 37.2 |
| ground | 466 | 4.02 [2.84, 5.93] | 5.96 | 0.165 | 0.238 | 0.366 | 59.8 | 46.4 |
| contrast_q1 | 134 | 2.27 [2, 4.16] | 5.71 | 0.194 | 0.328 | 0.322 | 57.1 | 42.2 |
| contrast_q2 | 139 | 4.26 [2.93, 5.93] | 5.97 | 0.158 | 0.209 | 0.371 | 61.2 | 46.6 |
| contrast_q3 | 163 | 4.64 [3.15, 6.53] | 6.41 | 0.153 | 0.19 | 0.367 | 68.2 | 54.6 |
| contrast_q4 | 107 | 3.58 [2.46, 4.96] | 5.07 | 0.168 | 0.262 | 0.366 | 45 | 32.2 |
| contrast_q5 | 85 | 4.14 [2.36, 5.32] | 5.42 | 0.224 | 0.282 | 0.337 | 49.2 | 37.4 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 82 | 1.05 | 3.04 [2.02, 4.13] | 0.327 |
| 3 | 84 | 1.1 | 3.04 [2.11, 4.12] | 0.326 |
| 5 | 85 | 1.05 | 3.01 [2.12, 4.09] | 0.323 |
| 9 | 85 | 1.05 | 3.01 [2.12, 4.09] | 0.323 |

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
| 2 | 95 | 0.99 | 1 | 0.5 | 0.99 [0.948, 1.2] | 0.975 |
| 3-4 | 228 | 0.915 | 1 | 0.284 | 0.915 [0.799, 1.25] | 0.864 |
| 5-8 | 230 | 0.784 | 1 | 0.155 | 0.784 [0.42, 1.22] | 0.787 |
| 9-16 | 275 | 0.779 | 1 | 0.0823 | 0.779 [0.493, 1.16] | 0.674 |
| 17+ | 624 | 0.503 | 1 | 0.0412 | 0.503 [0.341, 0.823] | 0.202 |

## Residual parallax on textured land (A15)

p90_px_scale1: 467, p90_px_scale2: 233, p90_px_scale4: 117, p90_px_scale8: 58.3, n_pixels: 3.5e+05

Runtime 6.8 min.
