# Stage A summary: matrixcity block3, source exact_independent, step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 0.982 [0.959, 1.02], pass.
- Noise model: white pixel noise sigma (global 0.03956) plus a record floor tau (global 0.0105, per-frame median 0.0105, p10-p90 0.002473-0.01621).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.04, 6-8 px 1.02, 9-184 px 0.853.

## Alignment (A5)

Mode metric, gauge k = 0.9754, beta = 0, last change 0.000435 (converged).
Block datum: calibration (factor 1.0257).
Median relative depth residual: label-free with parallax datum 0.0341, after datum 0.0301, oracle per-frame fit 0.03.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 7.26 m; additivity max relative difference 6.08e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 1.09 [1.05, 1.14] | 1.1 | 0.62 | 0.898 | 0.0567 | 2.93 | 0.0293 |
| land | 628 | 1.11 [1.05, 1.22] | 1.13 | 0.597 | 0.884 | 0.068 | 2.29 | 0.0465 |
| water | 501 | 1.07 [1.01, 1.13] | 1.07 | 0.649 | 0.916 | 0.0426 | 3.73 | 0.0128 |
| roof | 162 | 1.15 [1.05, 1.32] | 1.18 | 0.58 | 0.858 | 0.106 | 3.79 | 0.0946 |
| ground | 466 | 1.09 [1.04, 1.19] | 1.11 | 0.603 | 0.893 | 0.0556 | 1.77 | 0.0333 |
| contrast_q1 | 134 | 1.08 [1.04, 1.19] | 1.14 | 0.604 | 0.866 | 0.0687 | 3.97 | 0.0525 |
| contrast_q2 | 139 | 1.06 [0.961, 1.14] | 1.07 | 0.647 | 0.899 | 0.0329 | 2.83 | 0.014 |
| contrast_q3 | 163 | 1.14 [1.08, 1.28] | 1.14 | 0.558 | 0.89 | 0.0799 | 1.9 | 0.057 |
| contrast_q4 | 107 | 1.15 [1.01, 1.35] | 1.21 | 0.561 | 0.86 | 0.11 | 1.54 | 0.12 |
| contrast_q5 | 85 | 1.12 [1.01, 1.27] | 1.08 | 0.624 | 0.906 | 0.0588 | 0.434 | 0.0168 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 21 | 1.01 | 1.08 [1, 1.19] | 0.895 |
| 3 | 20 | 1.01 | 1.08 [1, 1.19] | 0.895 |
| 5 | 20 | 1.01 | 1.08 [1, 1.19] | 0.895 |
| 9 | 20 | 1.01 | 1.08 [1, 1.19] | 0.895 |

## Repetition (A10)

r(s), same strip: s=1: 0.0225, s=2: 0.00269, s=3: 0.00159, s=4: -0.00272, s=5: -0.00916, s=6: 0.0148, s=7: -0.00587, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00251; cross heading r = -0.00138.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.0103 [-0.00172, 0.0319] | 0.00519 [-0.00485, 0.011] |
| 1 | 0.00114 [-0.00317, 0.0111] | 0.00791 [-0.00535, 0.017] |
| 2 | -0.000389 [-0.00467, 0.00976] | 0.00882 [-0.00702, 0.0191] |
| 3 | -0.00318 [-0.0227, 0.00473] | 0.012 [-0.0075, 0.0272] |
| 5 | 0.0136 [-0.00697, 0.097] | -0.00555 [-0.0329, 0.0388] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = -0.00318 [-0.0227, 0.00473]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.494 | 0.5 | 0.5 | 0.987 [0.792, 1.05] | 0.0598 |
| 3-4 | 228 | 0.318 | 0.284 | 0.284 | 1.12 [0.957, 1.31] | 0.101 |
| 5-8 | 230 | 0.12 | 0.155 | 0.155 | 0.774 [0.601, 1.35] | 0.199 |
| 9-16 | 275 | 0.0873 | 0.0823 | 0.0823 | 1.06 [0.977, 1.28] | 0.0655 |
| 17+ | 624 | 0.0545 | 0.0412 | 0.0412 | 1.32 [0.95, 1.6] | 0.326 |

## Residual parallax on textured land (A15)

p90_px_scale1: 22.1, p90_px_scale2: 11, p90_px_scale4: 5.52, p90_px_scale8: 2.76, n_pixels: 3.5e+05

Runtime 6.6 min.
