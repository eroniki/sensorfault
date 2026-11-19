# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.16 [0.806, 1.44], FAIL (noise model at fault, H1 undefined).
- Noise model (class): white pixel noise sigma (global 0.0008411) plus a record floor tau (global 0.01789, per-frame median 0.01789, p10-p90 0.009027-0.04415); floor shape log g: roof +1.064, steep +0.460, log_pixels -0.038, log_incidence -0.084, log_depth +0.881 (g p10-p90 0.799-1.6).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.19, 6-8 px 1.07, 9-184 px 1.25; by class: roof 0.703, ground 1.28.

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: calibration (factor 0.98133).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.047, oracle per-frame fit 0.0338.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.154 m; additivity max relative difference 6.45e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 7.14 [4.05, 10.9] | 11.9 | 0.109 | 0.157 | 0.412 | 212 | 203 |
| land | 628 | 3.71 [2.16, 5.9] | 9.85 | 0.196 | 0.282 | 0.342 | 145 | 137 |
| water | 501 | 12.6 [7.67, 20.4] | 14 | 0 | 0 | 0.5 | 295 | 285 |
| roof | 162 | 3.01 [1.06, 4.21] | 5.04 | 0.265 | 0.346 | 0.273 | 40.2 | 31.8 |
| ground | 466 | 4.29 [2.4, 6.89] | 11 | 0.172 | 0.26 | 0.367 | 181 | 174 |
| contrast_q1 | 134 | 3.83 [2.26, 7.37] | 5.74 | 0.097 | 0.194 | 0.418 | 50.2 | 42.7 |
| contrast_q2 | 139 | 3.69 [1.99, 6.13] | 7.41 | 0.201 | 0.281 | 0.322 | 82.7 | 74.9 |
| contrast_q3 | 163 | 4.3 [1.95, 7.12] | 9.27 | 0.209 | 0.276 | 0.332 | 128 | 121 |
| contrast_q4 | 107 | 2.96 [1.33, 5.36] | 5.03 | 0.299 | 0.393 | 0.269 | 37.1 | 31.6 |
| contrast_q5 | 85 | 3.01 [2.4, 5.08] | 19.4 | 0.188 | 0.294 | 0.366 | 565 | 557 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 87 | 1.92 | 1.83 [1.28, 3.81] | 0.499 |
| 3 | 84 | 2.01 | 1.88 [1.33, 3.83] | 0.486 |
| 5 | 83 | 2.06 | 1.88 [1.33, 3.83] | 0.486 |
| 9 | 83 | 2.06 | 1.88 [1.33, 3.83] | 0.486 |

## Repetition (A10)

r(s), same strip: s=1: 0.897, s=2: 0.796, s=3: 0.699, s=4: 0.699, s=5: 0.87, s=6: 0.757, s=7: 0.258, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.77; cross heading r = 0.779.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.817 [0.699, 0.931] | 0.629 [0.474, 0.741] |
| 1 | 0.757 [0.617, 0.897] | 0.585 [0.454, 0.717] |
| 2 | 0.718 [0.558, 0.926] | 0.525 [0.422, 0.645] |
| 3 | 0.745 [0.5, 1.1] | 0.488 [0.379, 0.642] |
| 5 | 0.727 [0.238, 1.76] | 0.355 [-0.0319, 0.651] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.745 [0.5, 1.1]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.867 | 0.873 | 0.5 | 0.994 [0.813, 1.17] | 0.938 |
| 3-4 | 228 | 0.74 | 0.818 | 0.284 | 0.905 [0.707, 1.16] | 0.806 |
| 5-8 | 230 | 0.618 | 0.785 | 0.155 | 0.787 [0.521, 1.11] | 0.611 |
| 9-16 | 275 | 0.72 | 0.766 | 0.0823 | 0.94 [0.63, 1.32] | 0.923 |
| 17+ | 624 | 0.833 | 0.756 | 0.0412 | 1.1 [0.746, 1.62] | 1.1 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.9, p90_px_scale4: 9.43, p90_px_scale8: 4.72, n_pixels: 3.5e+05

Runtime 7.0 min.
