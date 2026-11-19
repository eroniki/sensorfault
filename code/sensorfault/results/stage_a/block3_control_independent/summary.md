# Stage A summary: matrixcity block3, source exact_independent, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 1 [0.992, 1.02], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9754, beta = 0, last change 0.000878 (converged).
Block datum: calibration (factor 1.0256).
Median relative depth residual: label-free with parallax datum 0.034, after datum 0.0302, oracle per-frame fit 0.03.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 7.26 m; additivity max relative difference 6.39e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 1.1 [1.06, 1.15] | 1.12 | 0.606 | 0.893 | 0.0644 | 2.91 | 0.0416 |
| land | 628 | 1.12 [1.05, 1.23] | 1.16 | 0.58 | 0.861 | 0.0809 | 2.27 | 0.076 |
| water | 501 | 1.07 [1.04, 1.13] | 1.06 | 0.639 | 0.932 | 0.0438 | 3.71 | 0.0118 |
| roof | 162 | 1.16 [1.08, 1.31] | 1.19 | 0.568 | 0.84 | 0.107 | 3.82 | 0.101 |
| ground | 466 | 1.11 [1.04, 1.23] | 1.15 | 0.584 | 0.869 | 0.0723 | 1.72 | 0.0682 |
| contrast_q1 | 134 | 1.11 [1.04, 1.23] | 1.15 | 0.59 | 0.858 | 0.0844 | 4.02 | 0.067 |
| contrast_q2 | 139 | 1.06 [0.974, 1.17] | 1.08 | 0.633 | 0.885 | 0.0443 | 2.84 | 0.0183 |
| contrast_q3 | 163 | 1.15 [1.11, 1.28] | 1.16 | 0.558 | 0.871 | 0.0954 | 1.8 | 0.0772 |
| contrast_q4 | 107 | 1.2 [0.995, 1.51] | 1.28 | 0.514 | 0.794 | 0.133 | 1.56 | 0.218 |
| contrast_q5 | 85 | 1.11 [1.01, 1.31] | 1.15 | 0.6 | 0.894 | 0.0641 | 0.352 | 0.0682 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 23 | 1.01 | 1.08 [1, 1.2] | 0.885 |
| 3 | 23 | 1 | 1.08 [1, 1.2] | 0.885 |
| 5 | 23 | 1.01 | 1.08 [1, 1.2] | 0.885 |
| 9 | 23 | 1.01 | 1.08 [1, 1.2] | 0.885 |

## Repetition (A10)

r(s), same strip: s=1: 0.0225, s=2: 0.00269, s=3: 0.0016, s=4: -0.00271, s=5: -0.00916, s=6: 0.0149, s=7: -0.00585, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00251; cross heading r = -0.00138.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.00343 [-0.00469, 0.0242] | 0.000154 [-0.0152, 0.00992] |
| 1 | 0.000252 [-0.00641, 0.0287] | -8.25e-05 [-0.0175, 0.0109] |
| 2 | -0.000235 [-0.00916, 0.034] | -0.0015 [-0.0209, 0.0106] |
| 3 | -0.000694 [-0.0156, 0.0411] | -0.00352 [-0.0246, 0.0118] |
| 5 | 0.00455 [-0.0163, 0.103] | -0.0159 [-0.0491, 0.0243] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = -0.000694 [-0.0156, 0.0411]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.494 | 0.5 | 0.5 | 0.987 [0.792, 1.04] | 0.0605 |
| 3-4 | 228 | 0.318 | 0.284 | 0.284 | 1.12 [0.939, 1.29] | 0.101 |
| 5-8 | 230 | 0.12 | 0.155 | 0.155 | 0.774 [0.576, 1.33] | 0.2 |
| 9-16 | 275 | 0.0873 | 0.0823 | 0.0823 | 1.06 [0.829, 1.26] | 0.0661 |
| 17+ | 624 | 0.0545 | 0.0412 | 0.0412 | 1.32 [0.605, 1.6] | 0.318 |

## Residual parallax on textured land (A15)

p90_px_scale1: 22.1, p90_px_scale2: 11, p90_px_scale4: 5.52, p90_px_scale8: 2.76, n_pixels: 3.5e+05

Runtime 8.0 min.
