# Stage A summary: matrixcity block3, source exact_independent, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **inconclusive**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 1 [0.99, 1.02], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9754, beta = 0, last change 0.000878 (converged).
Block datum: calibration (factor 1.0256).
Median relative depth residual: label-free with parallax datum 0.034, after datum 0.0302, oracle per-frame fit 0.03.

## Records (A7)

100690 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 7.25 m; additivity max relative difference 7.11e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 1.2 [1.11, 1.31] | 1.25 | 0.514 | 0.846 | 0.119 | 3.38 | 0.173 |
| land | 628 | 1.25 [1.13, 1.45] | 1.31 | 0.481 | 0.814 | 0.141 | 2.82 | 0.261 |
| water | 501 | 1.14 [1.04, 1.24] | 1.17 | 0.555 | 0.886 | 0.093 | 4.09 | 0.0826 |
| roof | 162 | 1.27 [1.1, 1.48] | 1.31 | 0.457 | 0.784 | 0.149 | 4.3 | 0.268 |
| ground | 466 | 1.24 [1.13, 1.43] | 1.31 | 0.489 | 0.824 | 0.137 | 2.3 | 0.259 |
| contrast_q1 | 134 | 1.25 [1.1, 1.48] | 1.26 | 0.478 | 0.821 | 0.141 | 4.42 | 0.189 |
| contrast_q2 | 139 | 1.13 [1.02, 1.31] | 1.18 | 0.561 | 0.878 | 0.0869 | 3.19 | 0.0876 |
| contrast_q3 | 163 | 1.31 [1.14, 1.48] | 1.33 | 0.448 | 0.798 | 0.164 | 2.41 | 0.29 |
| contrast_q4 | 107 | 1.34 [1.11, 1.64] | 1.49 | 0.402 | 0.738 | 0.192 | 2.47 | 0.642 |
| contrast_q5 | 85 | 1.2 [1.04, 1.53] | 1.3 | 0.518 | 0.824 | 0.119 | 0.919 | 0.255 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 29 | 1 | 1.19 [1.06, 1.42] | 0.81 |
| 3 | 30 | 1 | 1.19 [1.06, 1.41] | 0.806 |
| 5 | 30 | 1 | 1.19 [1.06, 1.41] | 0.806 |
| 9 | 30 | 1 | 1.19 [1.06, 1.41] | 0.806 |

## Repetition (A10)

r(s), same strip: s=1: 0.0249, s=2: 0.00459, s=3: 0.00292, s=4: -0.00248, s=5: -0.00736, s=6: 0.0125, s=7: -0.0109, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00305; cross heading r = -0.0018.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.00346 [-0.00557, 0.0233] | -0.00205 [-0.0181, 0.00963] |
| 1 | -0.000116 [-0.00761, 0.0268] | -0.00269 [-0.0212, 0.0107] |
| 2 | -0.00106 [-0.011, 0.0317] | -0.00458 [-0.0258, 0.00985] |
| 3 | -0.00205 [-0.0186, 0.0375] | -0.00744 [-0.03, 0.0106] |
| 5 | 0.000814 [-0.0226, 0.0902] | -0.0234 [-0.0651, 0.0226] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = -0.00205 [-0.0186, 0.0375]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.492 | 0.5 | 0.5 | 0.985 [0.797, 1.04] | 0.0782 |
| 3-4 | 228 | 0.317 | 0.284 | 0.284 | 1.11 [0.943, 1.3] | 0.105 |
| 5-8 | 230 | 0.125 | 0.155 | 0.155 | 0.81 [0.628, 1.33] | 0.196 |
| 9-16 | 275 | 0.0871 | 0.0823 | 0.0823 | 1.06 [0.853, 1.27] | 0.0899 |
| 17+ | 624 | 0.0546 | 0.0412 | 0.0412 | 1.33 [0.644, 1.59] | 0.367 |

## Residual parallax on textured land (A15)

p90_px_scale1: 22.1, p90_px_scale2: 11, p90_px_scale4: 5.52, p90_px_scale8: 2.76, n_pixels: 3.5e+05

Runtime 8.8 min.
