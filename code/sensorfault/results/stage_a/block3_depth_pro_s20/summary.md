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

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.154 m; additivity max relative difference 7.98e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 8.67 [5.27, 11.8] | 12.6 | 0.0833 | 0.131 | 0.436 | 237 | 227 |
| land | 628 | 5.45 [3.05, 7.9] | 10.2 | 0.15 | 0.231 | 0.384 | 156 | 148 |
| water | 501 | 13.1 [7.27, 22] | 15 | 0 | 0.00599 | 0.5 | 338 | 327 |
| roof | 162 | 7.46 [1.94, 11.3] | 11.4 | 0.142 | 0.21 | 0.386 | 197 | 187 |
| ground | 466 | 4.84 [2.69, 7.57] | 9.76 | 0.152 | 0.238 | 0.384 | 142 | 135 |
| contrast_q1 | 134 | 5.93 [2.53, 9.25] | 7.8 | 0.097 | 0.187 | 0.427 | 92.4 | 83.5 |
| contrast_q2 | 139 | 6.1 [3.32, 7.9] | 8.85 | 0.158 | 0.223 | 0.375 | 118 | 109 |
| contrast_q3 | 163 | 6.79 [2.76, 8.88] | 13.7 | 0.153 | 0.202 | 0.386 | 280 | 271 |
| contrast_q4 | 107 | 4.63 [1.75, 9.74] | 9.2 | 0.224 | 0.318 | 0.326 | 126 | 119 |
| contrast_q5 | 85 | 4.64 [2.65, 7.63] | 9.04 | 0.118 | 0.259 | 0.403 | 120 | 114 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 83 | 3.28 | 2.87 [1.88, 5.18] | 0.333 |
| 3 | 85 | 3.4 | 2.9 [1.88, 5.32] | 0.327 |
| 5 | 84 | 3.42 | 2.9 [1.94, 5.29] | 0.318 |
| 9 | 84 | 3.42 | 2.9 [1.94, 5.29] | 0.318 |

## Repetition (A10)

r(s), same strip: s=1: 0.897, s=2: 0.796, s=3: 0.699, s=4: 0.699, s=5: 0.87, s=6: 0.757, s=7: 0.258, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.77; cross heading r = 0.779.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.711 [0.553, 0.979] | 0.455 [0.364, 0.571] |
| 1 | 0.68 [0.501, 0.988] | 0.416 [0.331, 0.555] |
| 2 | 0.657 [0.443, 1.05] | 0.37 [0.27, 0.533] |
| 3 | 0.646 [0.405, 1.12] | 0.325 [0.205, 0.516] |
| 5 | 0.508 [0.187, 1.09] | 0.154 [-0.0228, 0.37] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.646 [0.405, 1.12]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.867 | 0.823 | 0.5 | 1.05 [0.832, 1.25] | 1.03 |
| 3-4 | 228 | 0.74 | 0.747 | 0.284 | 0.991 [0.722, 1.32] | 0.966 |
| 5-8 | 230 | 0.618 | 0.701 | 0.155 | 0.881 [0.544, 1.31] | 0.782 |
| 9-16 | 275 | 0.72 | 0.675 | 0.0823 | 1.07 [0.672, 1.6] | 1.11 |
| 17+ | 624 | 0.833 | 0.661 | 0.0412 | 1.26 [0.752, 1.99] | 1.33 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.9, p90_px_scale4: 9.43, p90_px_scale8: 4.72, n_pixels: 3.5e+05

Runtime 8.8 min.
