# Stage A summary: matrixcity block3, source cache (depth_pro), step 10

Frames 185 in 27 strips; entities 10235 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 2041, 'held_out': 2464, 'buffer': 5730}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.2 [0.948, 1.39], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 76.3, beta = 0, last change 0.000919 (converged).
Block datum: calibration (factor 0.96961).
Median relative depth residual: label-free with parallax datum 0.0513, after datum 0.0469, oracle per-frame fit 0.0338.

## Records (A7)

151131 records, 9617 entities seen, median n = 12, median pixels per record 20, median shape remainder 0.234 m; additivity max relative difference 8.11e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1666 | 7.36 [4, 10.9] | 11.7 | 0.128 | 0.194 | 0.401 | 205 | 196 |
| land | 1100 | 3.83 [2.43, 5.77] | 8.61 | 0.194 | 0.293 | 0.35 | 111 | 103 |
| water | 566 | 13 [10.9, 24.4] | 16.1 | 0 | 0.00177 | 0.5 | 388 | 377 |
| roof | 285 | 6.47 [1.63, 10.9] | 10.7 | 0.147 | 0.249 | 0.383 | 172 | 162 |
| ground | 815 | 3.41 [2.42, 5.08] | 7.77 | 0.21 | 0.308 | 0.339 | 90 | 82.9 |
| contrast_q1 | 217 | 4.61 [3.28, 7.52] | 7.69 | 0.0968 | 0.143 | 0.431 | 90.5 | 81 |
| contrast_q2 | 225 | 5.13 [1.94, 7.9] | 8.59 | 0.204 | 0.293 | 0.347 | 112 | 103 |
| contrast_q3 | 239 | 4.54 [2.2, 8.15] | 11 | 0.201 | 0.276 | 0.345 | 182 | 174 |
| contrast_q4 | 227 | 2.67 [1.69, 5] | 8.35 | 0.225 | 0.392 | 0.311 | 104 | 96.7 |
| contrast_q5 | 192 | 2.74 [1.82, 3.46] | 6.16 | 0.245 | 0.365 | 0.316 | 55 | 49.9 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 84 | 3.28 | 2.42 [1.68, 3.62] | 0.403 |
| 3 | 82 | 3.35 | 2.44 [1.68, 3.74] | 0.394 |
| 5 | 82 | 3.37 | 2.45 [1.69, 3.75] | 0.392 |
| 9 | 82 | 3.38 | 2.45 [1.69, 3.75] | 0.392 |

## Repetition (A10)

r(s), same strip: s=1: 0.534, s=2: 0.471, s=3: 0.411, s=4: 0.423, s=5: 0.558, s=6: 0.53, s=7: 0.145, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.438; cross heading r = 0.398.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.439 [0.198, 0.82] | 0.262 [0.125, 0.436] |
| 1 | 0.423 [0.189, 0.841] | 0.248 [0.122, 0.412] |
| 2 | 0.413 [0.163, 0.829] | 0.232 [0.114, 0.389] |
| 3 | 0.414 [0.164, 0.954] | 0.219 [0.0954, 0.383] |
| 5 | 0.337 [0.0748, 0.892] | 0.161 [0.0193, 0.315] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.414 [0.164, 0.954]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 136 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 185 | 0.59 | 0.707 | 0.5 | 0.835 [0.716, 1.06] | 0.231 |
| 3-4 | 329 | 0.325 | 0.585 | 0.293 | 0.555 [0.321, 1.15] | 0.594 |
| 5-8 | 380 | 0.201 | 0.505 | 0.156 | 0.397 [0.291, 1.06] | 0.169 |
| 9-16 | 340 | 0.521 | 0.464 | 0.0847 | 1.12 [0.58, 2.14] | 1.16 |
| 17+ | 946 | 0.763 | 0.437 | 0.0396 | 1.75 [0.815, 3.64] | 1.83 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.8, p90_px_scale4: 9.42, p90_px_scale8: 4.71, n_pixels: 1.48e+06

Runtime 11.1 min.
