# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.38 [1.08, 1.66], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.0008411) plus a record floor tau (global 0.01767, per-frame median 0.01767, p10-p90 0.008295-0.0438).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.44, 6-8 px 1.28, 9-184 px 1.48.

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: calibration (factor 0.98133).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.047, oracle per-frame fit 0.0338.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.154 m; additivity max relative difference 5.96e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 8.74 [5.44, 12.1] | 11.7 | 0.0717 | 0.122 | 0.442 | 204 | 196 |
| land | 628 | 5.43 [3.15, 7.65] | 8.2 | 0.129 | 0.217 | 0.396 | 99.5 | 93 |
| water | 501 | 13.8 [8.41, 22.1] | 14.9 | 0 | 0.00399 | 0.5 | 335 | 325 |
| roof | 162 | 6.86 [2.67, 9.6] | 9.41 | 0.0926 | 0.154 | 0.419 | 133 | 125 |
| ground | 466 | 4.92 [2.7, 7.52] | 7.73 | 0.142 | 0.238 | 0.388 | 88.1 | 82 |
| contrast_q1 | 134 | 5.99 [2.3, 9.57] | 7.18 | 0.0746 | 0.172 | 0.434 | 77.3 | 69.8 |
| contrast_q2 | 139 | 6.04 [3.61, 7.7] | 8.27 | 0.137 | 0.187 | 0.393 | 102 | 94.6 |
| contrast_q3 | 163 | 5.83 [3.11, 8.21] | 9.52 | 0.123 | 0.196 | 0.396 | 134 | 128 |
| contrast_q4 | 107 | 4.35 [1.77, 7.69] | 7.01 | 0.215 | 0.336 | 0.341 | 72 | 66.4 |
| contrast_q5 | 85 | 4.64 [3.26, 6.57] | 8.23 | 0.106 | 0.224 | 0.41 | 99 | 93.8 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 84 | 2.21 | 2.31 [1.57, 4.75] | 0.4 |
| 3 | 85 | 2.2 | 2.25 [1.5, 4.75] | 0.417 |
| 5 | 84 | 2.21 | 2.25 [1.5, 4.75] | 0.415 |
| 9 | 84 | 2.21 | 2.25 [1.5, 4.75] | 0.415 |

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
| 2 | 95 | 0.867 | 0.873 | 0.5 | 0.994 [0.813, 1.17] | 0.944 |
| 3-4 | 228 | 0.74 | 0.818 | 0.284 | 0.905 [0.707, 1.16] | 0.82 |
| 5-8 | 230 | 0.618 | 0.785 | 0.155 | 0.787 [0.521, 1.11] | 0.633 |
| 9-16 | 275 | 0.72 | 0.766 | 0.0823 | 0.94 [0.63, 1.32] | 0.938 |
| 17+ | 624 | 0.833 | 0.756 | 0.0412 | 1.1 [0.746, 1.62] | 1.15 |

## Residual parallax on textured land (A15)

p90_px_scale1: 37.7, p90_px_scale2: 18.9, p90_px_scale4: 9.43, p90_px_scale8: 4.72, n_pixels: 3.5e+05

Runtime 7.0 min.
