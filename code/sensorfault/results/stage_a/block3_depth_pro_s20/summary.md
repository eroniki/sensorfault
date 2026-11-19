# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.42 [1.11, 1.73], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: calibration (factor 0.98133).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.047, oracle per-frame fit 0.0338.

## Records (A7)

100688 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.154 m; additivity max relative difference 7.21e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 8.72 [5.22, 11.8] | 12.6 | 0.0824 | 0.131 | 0.436 | 237 | 228 |
| land | 628 | 5.48 [3.07, 7.88] | 10.3 | 0.148 | 0.231 | 0.384 | 157 | 149 |
| water | 501 | 13.1 [7.27, 22] | 15 | 0 | 0.00599 | 0.5 | 338 | 327 |
| roof | 162 | 7.36 [1.93, 11.4] | 11.4 | 0.142 | 0.21 | 0.386 | 196 | 186 |
| ground | 466 | 4.87 [2.76, 7.58] | 9.82 | 0.15 | 0.238 | 0.384 | 144 | 136 |
| contrast_q1 | 134 | 5.93 [2.51, 9.26] | 7.78 | 0.097 | 0.187 | 0.426 | 92 | 83.1 |
| contrast_q2 | 139 | 6.11 [3.31, 7.89] | 8.91 | 0.151 | 0.223 | 0.376 | 120 | 111 |
| contrast_q3 | 163 | 6.81 [2.72, 8.87] | 13.8 | 0.153 | 0.202 | 0.386 | 283 | 274 |
| contrast_q4 | 107 | 4.63 [1.77, 9.74] | 9.21 | 0.224 | 0.318 | 0.326 | 126 | 119 |
| contrast_q5 | 85 | 4.64 [2.85, 7.62] | 9.02 | 0.118 | 0.259 | 0.403 | 120 | 114 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 81 | 3.34 | 2.86 [1.91, 5.18] | 0.333 |
| 3 | 85 | 3.43 | 2.89 [1.88, 5.27] | 0.327 |
| 5 | 84 | 3.44 | 2.89 [1.91, 5.26] | 0.318 |
| 9 | 84 | 3.44 | 2.89 [1.91, 5.26] | 0.318 |

## Repetition (A10)

r(s), same strip: s=1: 0.897, s=2: 0.796, s=3: 0.699, s=4: 0.7, s=5: 0.87, s=6: 0.757, s=7: 0.259, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.771; cross heading r = 0.779.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.711 [0.553, 0.98] | 0.453 [0.364, 0.567] |
| 1 | 0.68 [0.501, 0.987] | 0.413 [0.329, 0.554] |
| 2 | 0.657 [0.443, 1.05] | 0.367 [0.27, 0.531] |
| 3 | 0.646 [0.405, 1.12] | 0.321 [0.205, 0.507] |
| 5 | 0.508 [0.187, 1.09] | 0.148 [-0.023, 0.351] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.646 [0.405, 1.12]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.867 | 0.823 | 0.5 | 1.05 [0.832, 1.25] | 1.04 |
| 3-4 | 228 | 0.747 | 0.747 | 0.284 | 1 [0.732, 1.33] | 0.983 |
| 5-8 | 230 | 0.618 | 0.701 | 0.155 | 0.881 [0.544, 1.3] | 0.781 |
| 9-16 | 275 | 0.72 | 0.675 | 0.0823 | 1.07 [0.672, 1.6] | 1.11 |
| 17+ | 624 | 0.833 | 0.661 | 0.0412 | 1.26 [0.752, 1.99] | 1.33 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.9, p90_px_scale4: 9.43, p90_px_scale8: 4.72, n_pixels: 3.5e+05

Runtime 8.7 min.
