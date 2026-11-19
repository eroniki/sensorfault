# Stage A summary: matrixcity block3, source cache (depth_pro), step 10

Frames 185 in 27 strips; entities 10235 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 2041, 'held_out': 2464, 'buffer': 5730}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.28 [1.05, 1.5], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.001015) plus a record floor tau (global 0.02001, per-frame median 0.02001, p10-p90 0.00941-0.0475).
- A8 s_z by pixels per record (held-out land): 4-11 px 1.46, 12-22 px 1.32, 23-762 px 1.05.

## Alignment (A5)

Mode metric, gauge k = 76.3, beta = 0, last change 0.000919 (converged).
Block datum: calibration (factor 0.96961).
Median relative depth residual: label-free with parallax datum 0.0513, after datum 0.0469, oracle per-frame fit 0.0338.

## Records (A7)

151131 records, 9617 entities seen, median n = 12, median pixels per record 20, median shape remainder 0.234 m; additivity max relative difference 7.66e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1666 | 7.06 [4.36, 9.99] | 11.8 | 0.0966 | 0.154 | 0.423 | 209 | 201 |
| land | 1100 | 4.51 [2.84, 6.5] | 10.9 | 0.146 | 0.233 | 0.383 | 175 | 168 |
| water | 566 | 12 [8.47, 20.3] | 13.6 | 0 | 0.00177 | 0.5 | 276 | 266 |
| roof | 285 | 6.22 [1.87, 9.47] | 8.31 | 0.133 | 0.232 | 0.392 | 104 | 95.8 |
| ground | 815 | 4 [2.91, 6.2] | 11.6 | 0.151 | 0.233 | 0.38 | 200 | 193 |
| contrast_q1 | 217 | 4.81 [2.85, 7.77] | 6.39 | 0.0922 | 0.157 | 0.425 | 61.7 | 54.2 |
| contrast_q2 | 225 | 5.5 [2.2, 8.06] | 15.9 | 0.187 | 0.258 | 0.359 | 377 | 367 |
| contrast_q3 | 239 | 4.63 [2.38, 7.48] | 8 | 0.121 | 0.234 | 0.391 | 94.2 | 88.4 |
| contrast_q4 | 227 | 3.57 [1.99, 6.23] | 12.4 | 0.185 | 0.282 | 0.357 | 229 | 222 |
| contrast_q5 | 192 | 4.27 [3.3, 5.34] | 8.49 | 0.146 | 0.229 | 0.385 | 104 | 100 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 86 | 2.63 | 2.22 [1.62, 3.85] | 0.413 |
| 3 | 88 | 2.61 | 2.3 [1.71, 4.11] | 0.399 |
| 5 | 88 | 2.61 | 2.3 [1.71, 4.11] | 0.399 |
| 9 | 88 | 2.61 | 2.3 [1.71, 4.11] | 0.399 |

## Repetition (A10)

r(s), same strip: s=1: 0.531, s=2: 0.469, s=3: 0.409, s=4: 0.422, s=5: 0.555, s=6: 0.527, s=7: 0.144, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.436; cross heading r = 0.396.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.484 [0.247, 0.783] | 0.32 [0.143, 0.56] |
| 1 | 0.45 [0.225, 0.746] | 0.301 [0.145, 0.53] |
| 2 | 0.432 [0.198, 0.736] | 0.278 [0.137, 0.476] |
| 3 | 0.463 [0.207, 0.887] | 0.272 [0.134, 0.466] |
| 5 | 0.485 [0.114, 1.28] | 0.237 [0.0376, 0.466] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.463 [0.207, 0.887]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 136 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 185 | 0.59 | 0.731 | 0.5 | 0.807 [0.713, 1] | 0.258 |
| 3-4 | 329 | 0.325 | 0.62 | 0.293 | 0.524 [0.329, 1.04] | 0.346 |
| 5-8 | 380 | 0.2 | 0.547 | 0.156 | 0.366 [0.281, 0.937] | 0.127 |
| 9-16 | 340 | 0.521 | 0.508 | 0.0847 | 1.02 [0.569, 1.79] | 0.857 |
| 17+ | 946 | 0.763 | 0.484 | 0.0396 | 1.58 [0.852, 2.89] | 1.63 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.8, p90_px_scale4: 9.42, p90_px_scale8: 4.71, n_pixels: 1.48e+06

Runtime 11.5 min.
