# Stage A summary: matrixcity block3, source exact_repeated, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.16 [1, 1.38], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.001814) plus a record floor tau (global 0.007289, per-frame median 0.007289, p10-p90 0.006538-0.008388).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.08, 6-8 px 1.08, 9-184 px 1.52.

## Alignment (A5)

Mode metric, gauge k = 0.9923, beta = 0, last change 0.224 (not below 0.1%).
Block datum: calibration (factor 1.0077).
Median relative depth residual: label-free with parallax datum 0.0101, after datum 0.00743, oracle per-frame fit 0.0074.

## Records (A7)

100686 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.33 m; additivity max relative difference 5.21e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 2.42 [2.16, 2.7] | 6.96 | 0.136 | 0.318 | 0.394 | 70 | 65.3 |
| land | 628 | 2.6 [2.2, 2.98] | 9.09 | 0.129 | 0.29 | 0.403 | 121 | 116 |
| water | 501 | 2.19 [1.92, 2.47] | 2.34 | 0.144 | 0.353 | 0.382 | 6.28 | 4.15 |
| roof | 162 | 2.47 [1.83, 2.86] | 2.93 | 0.13 | 0.34 | 0.394 | 10.7 | 8.14 |
| ground | 466 | 2.7 [2.27, 3.08] | 10.4 | 0.129 | 0.273 | 0.407 | 159 | 154 |
| contrast_q1 | 134 | 2.29 [2.12, 2.56] | 3.91 | 0.179 | 0.351 | 0.378 | 21 | 17.3 |
| contrast_q2 | 139 | 2.5 [1.94, 3.27] | 3.93 | 0.129 | 0.324 | 0.402 | 20.5 | 17.5 |
| contrast_q3 | 163 | 2.7 [2, 3.53] | 4.7 | 0.141 | 0.282 | 0.392 | 29.5 | 27 |
| contrast_q4 | 107 | 2.9 [2.09, 3.92] | 4.85 | 0.0841 | 0.243 | 0.437 | 31.5 | 29.1 |
| contrast_q5 | 85 | 3.33 [2.33, 4.06] | 22.1 | 0.0824 | 0.212 | 0.428 | 730 | 723 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 81 | 1.42 | 2.06 [1.84, 2.42] | 0.421 |
| 3 | 81 | 1.42 | 2.06 [1.84, 2.42] | 0.421 |
| 5 | 81 | 1.42 | 2.06 [1.84, 2.42] | 0.421 |
| 9 | 81 | 1.42 | 2.06 [1.84, 2.42] | 0.421 |

## Repetition (A10)

r(s), same strip: s=1: 0.23, s=2: 0.235, s=3: 0.231, s=4: 0.23, s=5: 0.223, s=6: 0.202, s=7: 0.254, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.224; cross heading r = 0.22.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.231 [0.182, 0.282] | 0.00624 [8.55e-06, 0.0132] |
| 1 | 0.232 [0.187, 0.284] | 0.00221 [-0.0057, 0.0137] |
| 2 | 0.229 [0.184, 0.282] | 0.00323 [-0.0076, 0.0153] |
| 3 | 0.226 [0.178, 0.287] | 0.00849 [-0.00871, 0.0348] |
| 5 | 0.205 [0.132, 0.314] | -0.0344 [-0.0687, 0.0464] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.226 [0.178, 0.287]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.673 | 0.613 | 0.5 | 1.1 [0.913, 1.24] | 1.05 |
| 3-4 | 228 | 0.403 | 0.446 | 0.284 | 0.903 [0.731, 1.1] | 0.882 |
| 5-8 | 230 | 0.362 | 0.346 | 0.155 | 1.05 [0.87, 1.22] | 0.994 |
| 9-16 | 275 | 0.221 | 0.29 | 0.0823 | 0.763 [0.582, 1.18] | 0.633 |
| 17+ | 624 | 0.291 | 0.258 | 0.0412 | 1.13 [0.976, 1.37] | 1.05 |

## Residual parallax on textured land (A15)

p90_px_scale1: 39.7, p90_px_scale2: 19.9, p90_px_scale4: 9.93, p90_px_scale8: 4.96, n_pixels: 3.5e+05

Runtime 7.2 min.
