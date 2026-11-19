# Stage A summary: matrixcity block3, source exact_repeated, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.18 [0.924, 1.57], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 0.9923, beta = 0, last change 0.224 (not below 0.1%).
Block datum: calibration (factor 1.0077).
Median relative depth residual: label-free with parallax datum 0.0101, after datum 0.00743, oracle per-frame fit 0.0074.

## Records (A7)

100686 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.33 m; additivity max relative difference 5.89e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 2.49 [2.17, 2.82] | 20.8 | 0.152 | 0.324 | 0.38 | 647 | 639 |
| land | 628 | 2.74 [2.24, 3.47] | 27.8 | 0.137 | 0.283 | 0.393 | 1.16e+03 | 1.15e+03 |
| water | 501 | 2.25 [1.89, 2.54] | 2.44 | 0.172 | 0.375 | 0.364 | 7.21 | 4.77 |
| roof | 162 | 2.54 [1.91, 3.38] | 3.53 | 0.154 | 0.309 | 0.376 | 16.8 | 13.4 |
| ground | 466 | 2.82 [2.32, 3.69] | 32.2 | 0.131 | 0.275 | 0.399 | 1.55e+03 | 1.54e+03 |
| contrast_q1 | 134 | 2.5 [2.13, 2.68] | 3.19 | 0.172 | 0.336 | 0.366 | 13.6 | 10.3 |
| contrast_q2 | 139 | 2.56 [2.08, 3.06] | 9.9 | 0.144 | 0.302 | 0.377 | 144 | 139 |
| contrast_q3 | 163 | 2.72 [1.97, 4.14] | 18 | 0.153 | 0.313 | 0.387 | 481 | 474 |
| contrast_q4 | 107 | 3.58 [2.69, 5.33] | 29.7 | 0.0935 | 0.196 | 0.43 | 1.32e+03 | 1.31e+03 |
| contrast_q5 | 85 | 3.79 [2.36, 5.88] | 61.7 | 0.0941 | 0.224 | 0.426 | 5.71e+03 | 5.7e+03 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 86 | 46.5 | 2.2 [1.87, 2.78] | 0.39 |
| 3 | 86 | 46.3 | 2.2 [1.87, 2.78] | 0.39 |
| 5 | 86 | 46.9 | 2.2 [1.87, 2.78] | 0.39 |
| 9 | 87 | 46.1 | 2.19 [1.86, 2.84] | 0.39 |

## Repetition (A10)

r(s), same strip: s=1: 0.23, s=2: 0.235, s=3: 0.231, s=4: 0.23, s=5: 0.223, s=6: 0.202, s=7: 0.254, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.224; cross heading r = 0.22.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.229 [0.184, 0.297] | -0.00707 [-0.0604, 0.00866] |
| 1 | 0.229 [0.187, 0.31] | -0.0102 [-0.0737, 0.00975] |
| 2 | 0.228 [0.182, 0.317] | -0.0125 [-0.0931, 0.0107] |
| 3 | 0.227 [0.181, 0.343] | -0.0154 [-0.113, 0.0184] |
| 5 | 0.228 [0.151, 0.426] | -0.0376 [-0.235, 0.00636] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.227 [0.181, 0.343]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.673 | 0.614 | 0.5 | 1.1 [0.905, 1.23] | 1.24 |
| 3-4 | 228 | 0.403 | 0.447 | 0.284 | 0.901 [0.716, 1.1] | 1.11 |
| 5-8 | 230 | 0.362 | 0.347 | 0.155 | 1.04 [0.798, 1.25] | 1.13 |
| 9-16 | 275 | 0.221 | 0.291 | 0.0823 | 0.76 [0.524, 1.24] | 0.705 |
| 17+ | 624 | 0.291 | 0.259 | 0.0412 | 1.12 [0.821, 1.4] | 1.15 |

## Residual parallax on textured land (A15)

p90_px_scale1: 39.7, p90_px_scale2: 19.9, p90_px_scale4: 9.93, p90_px_scale8: 4.96, n_pixels: 3.5e+05

Runtime 9.0 min.
