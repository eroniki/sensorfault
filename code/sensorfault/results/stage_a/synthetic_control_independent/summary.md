# Stage A summary: synthetic block3, source exact_independent, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 0.992 [0.957, 1.03], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9867, beta = 0, last change 0.000294 (converged).
Block datum: calibration (factor 1.0137).
Median relative depth residual: label-free with parallax datum 0.022, after datum 0.0214, oracle per-frame fit 0.0203.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 4.7 m; additivity max relative difference 4.1e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.02 [0.902, 1.14] | 1.01 | 0.645 | 0.965 | 0.0207 | 1.7 | 0.000562 |
| land | 274 | 1.02 [0.896, 1.14] | 1.01 | 0.65 | 0.967 | 0.0168 | 1.58 | 0.000112 |
| water | 123 | 1.03 [0.894, 1.19] | 1.03 | 0.634 | 0.959 | 0.0321 | 1.96 | 0.00276 |
| roof | 36 | 1.13 | 1.06 | 0.583 | 0.944 | 0.062 | 2.13 | 0.0116 |
| ground | 238 | 1.02 [0.896, 1.13] | 0.997 | 0.66 | 0.971 | 0.0123 | 1.5 | 2.23e-05 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 9 | 0.988 | 1.1 [0.848, 1.18] | 0.894 |
| 3 | 10 | 0.918 | 1.03 [0.831, 1.17] | 0.902 |
| 5 | 10 | 0.901 | 1.03 [0.831, 1.17] | 0.902 |
| 9 | 10 | 0.901 | 1.03 [0.831, 1.17] | 0.902 |

## Repetition (A10)

r(s), same strip: s=1: 0.0645, s=2: -0.0427, s=3: -0.0516, s=4: -0.0726, s=5: -0.0755, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.0151; cross heading r = 0.00922.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | -0.0356 [-0.0536, -0.00869] | -0.0552 [-0.0859, -0.033] |
| 1 | -0.0606 [-0.0795, -0.0253] | -0.0789 [-0.117, -0.0484] |
| 2 | -0.0666 [-0.0921, -0.00238] | -0.097 [-0.145, -0.0567] |
| 3 | -0.074 [-0.116, -0.00457] | -0.115 [-0.175, -0.0636] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = -0.074 [-0.116, -0.00457]

| n bin | entities | R_real | R_pred | 1/n | real/pred |
|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | n/a [n/a, n/a] |
| 2 | 172 | 0.409 | 0.5 | 0.5 | n/a [n/a, n/a] |
| 3-4 | 197 | 0.252 | 0.288 | 0.288 | n/a [n/a, n/a] |
| 5-8 | 146 | 0.127 | 0.152 | 0.152 | 0.84 [0.618, 1.02] |
| 9-16 | 236 | 0.0788 | 0.0882 | 0.0882 | 0.893 [0.761, 0.992] |
| 17+ | 15 | 0.0496 | 0.0564 | 0.0564 | 0.878 |

## Residual parallax on textured land (A15)

p90_px_scale1: 17.6, p90_px_scale2: 8.79, p90_px_scale4: 4.4, p90_px_scale8: 2.2, n_pixels: 1.95e+05

Runtime 0.4 min.
