# Stage A summary: matrixcity block3, source cache (metric3d_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.66 [1.19, 2.27], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.007297) plus a record floor tau (global 0.04135, per-frame median 0.04135, p10-p90 0.02821-0.07407).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.52, 6-8 px 1.44, 9-184 px 2.46.

## Alignment (A5)

Mode metric, gauge k = 6.61, beta = 0, last change 0.000515 (converged).
Block datum: calibration (factor 1.0814).
Median relative depth residual: label-free with parallax datum 0.159, after datum 0.157, oracle per-frame fit 0.113.

## Records (A7)

100690 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.931 m; additivity max relative difference 5.23e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 12 [7.08, 19.9] | 17.2 | 0.0531 | 0.0957 | 0.462 | 445 | 433 |
| land | 628 | 5.05 [3.67, 6.81] | 7.51 | 0.0955 | 0.172 | 0.431 | 86.7 | 77 |
| water | 501 | 24.2 [18.4, 33.1] | 24.4 | 0 | 0 | 0.5 | 894 | 881 |
| roof | 162 | 5.22 [3.1, 11.6] | 8.68 | 0.0802 | 0.142 | 0.441 | 117 | 105 |
| ground | 466 | 5.02 [3.84, 6.43] | 7.06 | 0.101 | 0.182 | 0.428 | 76.3 | 67.3 |
| contrast_q1 | 134 | 4.68 [2.51, 7.75] | 6.18 | 0.134 | 0.216 | 0.401 | 60.5 | 50.3 |
| contrast_q2 | 139 | 4.11 [2.95, 7.28] | 6.66 | 0.0647 | 0.158 | 0.449 | 69.1 | 59.3 |
| contrast_q3 | 163 | 5.25 [3.52, 6.76] | 8.05 | 0.0613 | 0.141 | 0.452 | 99 | 89.4 |
| contrast_q4 | 107 | 5.88 [4.76, 7.67] | 8.07 | 0.131 | 0.178 | 0.421 | 99.4 | 89.9 |
| contrast_q5 | 85 | 4.75 [3.2, 6.47] | 8.82 | 0.106 | 0.176 | 0.425 | 117 | 109 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 82 | 2.05 | 3.68 [2.8, 4.72] | 0.218 |
| 3 | 82 | 2.08 | 3.68 [2.8, 4.72] | 0.218 |
| 5 | 83 | 2.05 | 3.69 [2.8, 4.72] | 0.214 |
| 9 | 83 | 2.04 | 3.69 [2.8, 4.72] | 0.214 |

## Repetition (A10)

r(s), same strip: s=1: 0.889, s=2: 0.825, s=3: 0.702, s=4: 0.498, s=5: 0.265, s=6: 0.152, s=7: 0.0333, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.802; cross heading r = 0.712.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.783 [0.675, 0.896] | 0.495 [0.392, 0.566] |
| 1 | 0.703 [0.579, 0.834] | 0.44 [0.356, 0.504] |
| 2 | 0.582 [0.452, 0.77] | 0.357 [0.287, 0.412] |
| 3 | 0.412 [0.282, 0.621] | 0.257 [0.178, 0.327] |
| 5 | 0.145 [0.0658, 0.376] | 0.238 [-0.238, 0.388] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.412 [0.282, 0.621]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.926 | 0.706 | 0.5 | 1.31 [1.13, 1.45] | 1.24 |
| 3-4 | 228 | 0.773 | 0.579 | 0.284 | 1.33 [0.825, 1.8] | 1.32 |
| 5-8 | 230 | 0.863 | 0.503 | 0.155 | 1.72 [1.26, 2.24] | 1.65 |
| 9-16 | 275 | 0.772 | 0.461 | 0.0823 | 1.68 [1.04, 2.43] | 1.59 |
| 17+ | 624 | 0.893 | 0.436 | 0.0412 | 2.05 [1.36, 2.95] | 2.11 |

## Residual parallax on textured land (A15)

p90_px_scale1: 110, p90_px_scale2: 55, p90_px_scale4: 27.5, p90_px_scale8: 13.8, n_pixels: 3.5e+05

Runtime 7.2 min.
