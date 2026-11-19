# Stage A summary: matrixcity block3, source cache (depth_pro), step 10

Frames 185 in 27 strips; entities 10235 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 2041, 'held_out': 2464, 'buffer': 5730}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.1 [0.863, 1.29], pass.
- Noise model (class): white pixel noise sigma (global 0.001015) plus a record floor tau (global 0.02016, per-frame median 0.02016, p10-p90 0.009669-0.04684); floor shape log g: roof +0.978, steep +0.304, log_pixels -0.046, log_incidence -0.185, log_depth +1.031 (g p10-p90 0.745-1.69).
- A8 s_z by pixels per record (held-out land): 4-11 px 1.09, 12-22 px 1.18, 23-762 px 1.02; by class: roof 0.743, ground 1.19.

## Alignment (A5)

Mode metric, gauge k = 76.3, beta = 0, last change 0.000919 (converged).
Block datum: calibration (factor 0.96961).
Median relative depth residual: label-free with parallax datum 0.0513, after datum 0.0469, oracle per-frame fit 0.0338.

## Records (A7)

151131 records, 9617 entities seen, median n = 12, median pixels per record 20, median shape remainder 0.234 m; additivity max relative difference 5.95e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1666 | 5.97 [4.03, 9.29] | 23.3 | 0.128 | 0.19 | 0.397 | 811 | 801 |
| land | 1100 | 3.82 [2.35, 5.01] | 26.9 | 0.195 | 0.287 | 0.343 | 1.08e+03 | 1.07e+03 |
| water | 566 | 11.8 [9.45, 20.4] | 13.6 | 0 | 0.00177 | 0.5 | 280 | 270 |
| roof | 285 | 3.65 [0.909, 5.2] | 5.34 | 0.267 | 0.333 | 0.279 | 44.6 | 36.3 |
| ground | 815 | 3.88 [2.73, 5.34] | 31.1 | 0.169 | 0.271 | 0.366 | 1.45e+03 | 1.44e+03 |
| contrast_q1 | 217 | 4.06 [2.78, 5.73] | 5.35 | 0.0968 | 0.189 | 0.419 | 43.8 | 36.4 |
| contrast_q2 | 225 | 4.01 [1.95, 5.39] | 54.2 | 0.213 | 0.289 | 0.317 | 4.41e+03 | 4.39e+03 |
| contrast_q3 | 239 | 4.4 [1.94, 6.56] | 14.7 | 0.23 | 0.285 | 0.32 | 321 | 313 |
| contrast_q4 | 227 | 2.93 [1.49, 4.46] | 15.7 | 0.242 | 0.366 | 0.305 | 367 | 359 |
| contrast_q5 | 192 | 3.64 [2.74, 4.14] | 10.7 | 0.182 | 0.307 | 0.364 | 169 | 163 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 91 | 2.31 | 1.68 [1.28, 2.56] | 0.532 |
| 3 | 87 | 2.61 | 1.92 [1.4, 2.92] | 0.478 |
| 5 | 88 | 2.59 | 1.79 [1.31, 2.66] | 0.508 |
| 9 | 88 | 2.59 | 1.79 [1.31, 2.66] | 0.508 |

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
| 2 | 185 | 0.59 | 0.731 | 0.5 | 0.807 [0.713, 1] | 0.255 |
| 3-4 | 329 | 0.325 | 0.62 | 0.293 | 0.524 [0.329, 1.04] | 0.38 |
| 5-8 | 380 | 0.2 | 0.547 | 0.156 | 0.366 [0.281, 0.937] | 0.143 |
| 9-16 | 340 | 0.521 | 0.508 | 0.0847 | 1.02 [0.569, 1.79] | 0.907 |
| 17+ | 946 | 0.763 | 0.484 | 0.0396 | 1.58 [0.852, 2.89] | 1.57 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.8, p90_px_scale4: 9.42, p90_px_scale8: 4.71, n_pixels: 1.48e+06

Runtime 11.5 min.
