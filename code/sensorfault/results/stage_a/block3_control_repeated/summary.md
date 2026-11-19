# Stage A summary: matrixcity block3, source exact_repeated, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.05 [1, 1.09], pass.
- Noise model (class): white pixel noise sigma (global 0.001814) plus a record floor tau (global 0.007643, per-frame median 0.007643, p10-p90 0.007005-0.00816); floor shape log g: roof +0.069, steep +0.372, log_pixels +0.021, log_incidence -0.268, log_depth -0.668 (g p10-p90 0.829-1.56).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.03, 6-8 px 1.03, 9-184 px 1.09; by class: roof 1.03, ground 1.05.

## Alignment (A5)

Mode metric, gauge k = 0.9923, beta = 0, last change 0.224 (not below 0.1%).
Block datum: calibration (factor 1.0077).
Median relative depth residual: label-free with parallax datum 0.0101, after datum 0.00743, oracle per-frame fit 0.0074.

## Records (A7)

100686 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.33 m; additivity max relative difference 5.09e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 2.39 [2.2, 2.57] | 2.63 | 0.143 | 0.33 | 0.389 | 8.01 | 5.97 |
| land | 628 | 2.36 [2.04, 2.6] | 2.68 | 0.164 | 0.352 | 0.376 | 8.28 | 6.31 |
| water | 501 | 2.41 [2.11, 2.72] | 2.57 | 0.116 | 0.303 | 0.404 | 7.68 | 5.56 |
| roof | 162 | 2.22 [1.79, 2.57] | 2.43 | 0.167 | 0.377 | 0.378 | 7.02 | 4.67 |
| ground | 466 | 2.41 [2.13, 2.66] | 2.76 | 0.163 | 0.343 | 0.376 | 8.71 | 6.89 |
| contrast_q1 | 134 | 2.36 [2.25, 2.49] | 2.75 | 0.149 | 0.336 | 0.385 | 9.47 | 6.83 |
| contrast_q2 | 139 | 2.42 [1.84, 2.69] | 2.69 | 0.158 | 0.36 | 0.38 | 8.58 | 6.37 |
| contrast_q3 | 163 | 2.52 [1.93, 2.7] | 2.61 | 0.172 | 0.325 | 0.371 | 7.42 | 5.82 |
| contrast_q4 | 107 | 2.17 [1.75, 2.67] | 2.54 | 0.196 | 0.421 | 0.366 | 7.12 | 5.4 |
| contrast_q5 | 85 | 2.37 [1.89, 2.58] | 2.84 | 0.141 | 0.329 | 0.382 | 8.99 | 7.48 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 51 | 0.845 | 1.95 [1.8, 2.12] | 0.442 |
| 3 | 52 | 0.841 | 1.96 [1.8, 2.12] | 0.442 |
| 5 | 51 | 0.843 | 1.97 [1.82, 2.12] | 0.435 |
| 9 | 51 | 0.843 | 1.97 [1.82, 2.12] | 0.435 |

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
| 3-4 | 228 | 0.403 | 0.446 | 0.284 | 0.903 [0.731, 1.1] | 0.876 |
| 5-8 | 230 | 0.362 | 0.346 | 0.155 | 1.05 [0.87, 1.22] | 0.971 |
| 9-16 | 275 | 0.221 | 0.29 | 0.0823 | 0.763 [0.582, 1.18] | 0.634 |
| 17+ | 624 | 0.291 | 0.258 | 0.0412 | 1.13 [0.976, 1.37] | 1.06 |

## Residual parallax on textured land (A15)

p90_px_scale1: 39.7, p90_px_scale2: 19.9, p90_px_scale4: 9.93, p90_px_scale8: 4.96, n_pixels: 3.5e+05

Runtime 7.2 min.
