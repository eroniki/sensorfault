# Stage A summary: matrixcity block3, source cache (unidepth_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.37 [0.943, 2.06], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 1.112, beta = 0, last change 0.000309 (converged).
Block datum: calibration (factor 1.0142).
Median relative depth residual: label-free with parallax datum 0.0139, after datum 0.0064, oracle per-frame fit 0.00638.

## Records (A7)

100688 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.283 m; additivity max relative difference 8.32e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 4.51 [2.83, 7.7] | 10.8 | 0.132 | 0.207 | 0.401 | 170 | 166 |
| land | 628 | 3.98 [2.44, 5.71] | 12.6 | 0.167 | 0.242 | 0.373 | 234 | 230 |
| water | 501 | 5.04 [2.5, 12.3] | 7.93 | 0.0878 | 0.164 | 0.435 | 90.5 | 86.6 |
| roof | 162 | 7.12 [4.98, 9.45] | 17.9 | 0.0123 | 0.0309 | 0.489 | 477 | 471 |
| ground | 466 | 3.1 [1.65, 4.5] | 10.2 | 0.221 | 0.315 | 0.333 | 149 | 147 |
| contrast_q1 | 134 | 4 [2.02, 5.37] | 9.56 | 0.164 | 0.209 | 0.386 | 133 | 129 |
| contrast_q2 | 139 | 3.85 [2.54, 5.37] | 10.2 | 0.151 | 0.223 | 0.38 | 152 | 148 |
| contrast_q3 | 163 | 3.58 [1.82, 7.17] | 14.1 | 0.19 | 0.27 | 0.353 | 293 | 290 |
| contrast_q4 | 107 | 3.74 [2.43, 6.96] | 15.3 | 0.159 | 0.234 | 0.381 | 345 | 341 |
| contrast_q5 | 85 | 4.62 [1.61, 7.56] | 13.6 | 0.165 | 0.282 | 0.372 | 271 | 269 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 89 | 6.48 | 2.39 [1.59, 3.22] | 0.408 |
| 3 | 90 | 6.26 | 2.31 [1.41, 3.23] | 0.429 |
| 5 | 90 | 6.3 | 2.31 [1.41, 3.23] | 0.429 |
| 9 | 90 | 6.32 | 2.31 [1.41, 3.23] | 0.429 |

## Repetition (A10)

r(s), same strip: s=1: 0.454, s=2: 0.291, s=3: 0.196, s=4: 0.144, s=5: 0.184, s=6: 0.196, s=7: 0.155, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.191; cross heading r = 0.221.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.231 [0.104, 0.398] | 0.0804 [0.0481, 0.149] |
| 1 | 0.194 [0.0568, 0.391] | 0.0691 [0.039, 0.142] |
| 2 | 0.175 [-0.0275, 0.405] | 0.0586 [0.0315, 0.125] |
| 3 | 0.17 [-0.0993, 0.453] | 0.0519 [0.0289, 0.127] |
| 5 | 0.176 [-0.381, 0.686] | 0.0245 [0.0161, 0.0999] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.17 [-0.0993, 0.453]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.587 | 0.585 | 0.5 | 1 [0.838, 1.42] | 2.57 |
| 3-4 | 228 | 0.488 | 0.406 | 0.284 | 1.2 [0.839, 1.88] | 1.21 |
| 5-8 | 230 | 0.26 | 0.298 | 0.155 | 0.871 [0.382, 1.86] | 1.14 |
| 9-16 | 275 | 0.314 | 0.238 | 0.0823 | 1.32 [0.689, 4.41] | 1.93 |
| 17+ | 624 | 0.313 | 0.204 | 0.0412 | 1.53 [0.651, 8.33] | 1.35 |

## Residual parallax on textured land (A15)

p90_px_scale1: 10.7, p90_px_scale2: 5.37, p90_px_scale4: 2.69, p90_px_scale8: 1.34, n_pixels: 3.5e+05

Runtime 8.0 min.
