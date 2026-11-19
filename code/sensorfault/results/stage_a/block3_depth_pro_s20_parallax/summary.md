# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.2 [0.933, 1.48], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.0008394) plus a record floor tau (global 0.01837, per-frame median 0.01837, p10-p90 0.00711-0.05189).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.4, 6-8 px 1.1, 9-184 px 1.13.

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: parallax (factor 1).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.0495, oracle per-frame fit 0.0338.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.157 m; additivity max relative difference 5.69e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 9.03 [4.22, 12.7] | 16.1 | 0.0939 | 0.173 | 0.426 | 390 | 381 |
| land | 628 | 4.14 [2.51, 7.28] | 7.93 | 0.139 | 0.237 | 0.392 | 93.7 | 86.7 |
| water | 501 | 20.9 [9.82, 33.5] | 22.5 | 0.0379 | 0.0918 | 0.468 | 762 | 751 |
| roof | 162 | 2.92 [2.12, 6.33] | 5.23 | 0.167 | 0.235 | 0.373 | 41.7 | 34.5 |
| ground | 466 | 4.49 [2.61, 8.46] | 8.68 | 0.129 | 0.238 | 0.399 | 112 | 105 |
| contrast_q1 | 134 | 3.41 [2.63, 11] | 8.59 | 0.127 | 0.201 | 0.394 | 111 | 103 |
| contrast_q2 | 139 | 4.4 [2.21, 8.04] | 7.23 | 0.122 | 0.237 | 0.407 | 78.3 | 71 |
| contrast_q3 | 163 | 4.69 [2.16, 7.16] | 8.94 | 0.129 | 0.227 | 0.397 | 119 | 112 |
| contrast_q4 | 107 | 4.13 [2.14, 6.72] | 6.57 | 0.187 | 0.318 | 0.368 | 63.6 | 57.6 |
| contrast_q5 | 85 | 3.72 [2.33, 7.28] | 7.48 | 0.141 | 0.212 | 0.387 | 81.8 | 76.3 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 87 | 1.78 | 2.42 [1.79, 5.01] | 0.386 |
| 3 | 87 | 1.88 | 2.42 [1.79, 5.01] | 0.386 |
| 5 | 88 | 1.86 | 2.49 [1.8, 5.01] | 0.376 |
| 9 | 88 | 1.86 | 2.49 [1.8, 5.01] | 0.376 |

## Repetition (A10)

r(s), same strip: s=1: 0.93, s=2: 0.862, s=3: 0.797, s=4: 0.813, s=5: 0.978, s=6: 0.869, s=7: 0.454, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.848; cross heading r = 0.847.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.88 [0.785, 0.988] | 0.725 [0.613, 0.814] |
| 1 | 0.842 [0.727, 0.983] | 0.691 [0.591, 0.797] |
| 2 | 0.822 [0.672, 1.05] | 0.641 [0.543, 0.756] |
| 3 | 0.858 [0.624, 1.23] | 0.611 [0.488, 0.784] |
| 5 | 0.845 [0.262, 1.86] | 0.484 [0.00881, 0.804] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.858 [0.624, 1.23]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.878 | 0.929 | 0.5 | 0.945 [0.811, 1.11] | 0.911 |
| 3-4 | 228 | 0.785 | 0.898 | 0.284 | 0.873 [0.73, 1.1] | 0.794 |
| 5-8 | 230 | 0.67 | 0.88 | 0.155 | 0.761 [0.518, 1.04] | 0.689 |
| 9-16 | 275 | 0.779 | 0.87 | 0.0823 | 0.896 [0.641, 1.2] | 1.02 |
| 17+ | 624 | 0.885 | 0.864 | 0.0412 | 1.02 [0.788, 1.42] | 1.13 |

## Residual parallax on textured land (A15)

p90_px_scale1: 34, p90_px_scale2: 17, p90_px_scale4: 8.49, p90_px_scale8: 4.24, n_pixels: 3.5e+05

Runtime 7.1 min.
