# Stage A summary: matrixcity block3, source cache (depth_pro), step 10

Frames 185 in 27 strips; entities 10235 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 2041, 'held_out': 2464, 'buffer': 5730}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.2 [0.95, 1.39], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 76.3, beta = 0, last change 0.000919 (converged).
Block datum: calibration (factor 0.96961).
Median relative depth residual: label-free with parallax datum 0.0513, after datum 0.0469, oracle per-frame fit 0.0338.

## Records (A7)

151131 records, 9617 entities seen, median n = 12, median pixels per record 20, median shape remainder 0.234 m; additivity max relative difference 8.03e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1666 | 7.34 [3.98, 10.9] | 11.7 | 0.127 | 0.194 | 0.402 | 205 | 196 |
| land | 1100 | 3.83 [2.42, 5.77] | 8.59 | 0.193 | 0.293 | 0.352 | 111 | 103 |
| water | 566 | 13 [10.9, 24.5] | 16.1 | 0 | 0.00177 | 0.5 | 390 | 379 |
| roof | 285 | 6.46 [1.63, 10.9] | 10.6 | 0.147 | 0.249 | 0.383 | 171 | 161 |
| ground | 815 | 3.41 [2.39, 5.06] | 7.75 | 0.209 | 0.308 | 0.341 | 89.6 | 82.5 |
| contrast_q1 | 217 | 4.62 [3.28, 7.54] | 7.67 | 0.0968 | 0.143 | 0.431 | 90.1 | 80.7 |
| contrast_q2 | 225 | 5.13 [1.94, 7.73] | 8.57 | 0.204 | 0.289 | 0.348 | 112 | 102 |
| contrast_q3 | 239 | 4.54 [2.21, 8.14] | 11 | 0.197 | 0.28 | 0.349 | 180 | 172 |
| contrast_q4 | 227 | 2.68 [1.69, 4.98] | 8.33 | 0.225 | 0.392 | 0.311 | 103 | 96.3 |
| contrast_q5 | 192 | 2.74 [1.81, 3.46] | 6.14 | 0.245 | 0.365 | 0.318 | 54.7 | 49.6 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 84 | 3.26 | 2.39 [1.66, 3.59] | 0.41 |
| 3 | 84 | 3.35 | 2.43 [1.66, 3.75] | 0.409 |
| 5 | 84 | 3.36 | 2.43 [1.68, 3.55] | 0.395 |
| 9 | 83 | 3.37 | 2.45 [1.68, 3.75] | 0.394 |

## Repetition (A10)

r(s), same strip: s=1: 0.531, s=2: 0.469, s=3: 0.409, s=4: 0.422, s=5: 0.555, s=6: 0.527, s=7: 0.144, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.436; cross heading r = 0.396.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.437 [0.197, 0.82] | 0.261 [0.124, 0.437] |
| 1 | 0.421 [0.188, 0.841] | 0.247 [0.121, 0.412] |
| 2 | 0.411 [0.161, 0.829] | 0.231 [0.113, 0.389] |
| 3 | 0.412 [0.162, 0.954] | 0.218 [0.0946, 0.383] |
| 5 | 0.336 [0.074, 0.892] | 0.16 [0.0191, 0.315] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.412 [0.162, 0.954]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 136 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 185 | 0.59 | 0.706 | 0.5 | 0.836 [0.716, 1.06] | 0.23 |
| 3-4 | 329 | 0.325 | 0.584 | 0.293 | 0.556 [0.321, 1.15] | 0.596 |
| 5-8 | 380 | 0.2 | 0.504 | 0.156 | 0.397 [0.292, 1.06] | 0.167 |
| 9-16 | 340 | 0.521 | 0.462 | 0.0847 | 1.13 [0.581, 2.15] | 1.17 |
| 17+ | 946 | 0.763 | 0.435 | 0.0396 | 1.75 [0.815, 3.67] | 1.83 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.8, p90_px_scale4: 9.42, p90_px_scale8: 4.71, n_pixels: 1.48e+06

Runtime 10.8 min.
