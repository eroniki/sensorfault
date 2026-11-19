# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.24 [0.98, 1.47], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: parallax (factor 1).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.0495, oracle per-frame fit 0.0338.

## Records (A7)

100688 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.157 m; additivity max relative difference 6.94e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 8.47 [4.54, 11.7] | 16.9 | 0.0753 | 0.154 | 0.44 | 430 | 419 |
| land | 628 | 4.38 [2.93, 7.12] | 7.94 | 0.116 | 0.21 | 0.411 | 94.8 | 86.9 |
| water | 501 | 18.6 [8.71, 35.4] | 23.8 | 0.024 | 0.0838 | 0.476 | 850 | 838 |
| roof | 162 | 3.47 [2.3, 6.51] | 6.21 | 0.123 | 0.216 | 0.406 | 60 | 50.9 |
| ground | 466 | 4.73 [3.04, 8.27] | 8.46 | 0.114 | 0.208 | 0.413 | 107 | 99.4 |
| contrast_q1 | 134 | 3.36 [2.69, 9.45] | 8.79 | 0.112 | 0.201 | 0.405 | 118 | 108 |
| contrast_q2 | 139 | 4.48 [2.56, 7.16] | 6.95 | 0.0935 | 0.201 | 0.423 | 73.6 | 65.1 |
| contrast_q3 | 163 | 5.2 [2.26, 7.89] | 8.52 | 0.129 | 0.227 | 0.415 | 109 | 101 |
| contrast_q4 | 107 | 4.26 [2.86, 7.87] | 7.88 | 0.121 | 0.224 | 0.398 | 92.5 | 85.5 |
| contrast_q5 | 85 | 4.28 [2.89, 6.79] | 6.9 | 0.129 | 0.188 | 0.415 | 70.1 | 64.2 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 83 | 2.78 | 3.04 [2.17, 5.5] | 0.29 |
| 3 | 82 | 3.07 | 3.06 [2.24, 5.5] | 0.288 |
| 5 | 83 | 3.14 | 3.06 [2.24, 5.5] | 0.288 |
| 9 | 83 | 3.17 | 3.06 [2.24, 5.5] | 0.288 |

## Repetition (A10)

r(s), same strip: s=1: 0.931, s=2: 0.862, s=3: 0.797, s=4: 0.813, s=5: 0.979, s=6: 0.87, s=7: 0.454, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.849; cross heading r = 0.848.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.815 [0.64, 1.1] | 0.562 [0.46, 0.691] |
| 1 | 0.796 [0.597, 1.12] | 0.527 [0.421, 0.671] |
| 2 | 0.782 [0.528, 1.2] | 0.484 [0.346, 0.685] |
| 3 | 0.779 [0.486, 1.27] | 0.439 [0.271, 0.657] |
| 5 | 0.662 [0.159, 1.27] | 0.26 [0.0135, 0.495] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.779 [0.486, 1.27]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.878 | 0.889 | 0.5 | 0.987 [0.825, 1.2] | 0.98 |
| 3-4 | 228 | 0.791 | 0.842 | 0.284 | 0.939 [0.75, 1.26] | 0.906 |
| 5-8 | 230 | 0.67 | 0.813 | 0.155 | 0.824 [0.539, 1.2] | 0.798 |
| 9-16 | 275 | 0.779 | 0.797 | 0.0823 | 0.977 [0.68, 1.46] | 1.15 |
| 17+ | 624 | 0.885 | 0.788 | 0.0412 | 1.12 [0.794, 1.73] | 1.23 |

## Residual parallax on textured land (A15)

p90_px_scale1: 34, p90_px_scale2: 17, p90_px_scale4: 8.49, p90_px_scale8: 4.24, n_pixels: 3.5e+05

Runtime 8.7 min.
