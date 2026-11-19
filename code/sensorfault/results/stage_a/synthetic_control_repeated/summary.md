# Stage A summary: synthetic block3, source exact_repeated, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.01 [0.992, 1.05], pass.

## Alignment (A5)

Mode metric, gauge k = 0.996, beta = 0, last change 0.000266 (converged).
Block datum: calibration (factor 1.0048).
Median relative depth residual: label-free with parallax datum 0.00845, after datum 0.00819, oracle per-frame fit 0.00784.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.313 m; additivity max relative difference 5.26e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 2.01 [1.81, 2.13] | 2.17 | 0.219 | 0.441 | 0.336 | 5.9 | 3.24 |
| land | 274 | 2.06 [1.85, 2.21] | 2.26 | 0.212 | 0.423 | 0.339 | 6.37 | 3.71 |
| water | 123 | 1.87 [1.67, 2.11] | 1.96 | 0.236 | 0.48 | 0.33 | 4.85 | 2.23 |
| roof | 36 | 1.92 | 2.13 | 0.167 | 0.472 | 0.339 | 5.87 | 3.06 |
| ground | 238 | 2.1 [1.85, 2.27] | 2.28 | 0.218 | 0.416 | 0.339 | 6.44 | 3.8 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 9 | 0.922 | 1.98 [1.85, 2.06] | 0.444 |
| 3 | 7 | 0.911 | 2.01 [1.85, 2.08] | 0.441 |
| 5 | 7 | 0.913 | 2.01 [1.85, 2.08] | 0.441 |
| 9 | 7 | 0.913 | 2.01 [1.85, 2.08] | 0.441 |

## Repetition (A10)

r(s), same strip: s=1: 0.309, s=2: 0.27, s=3: 0.232, s=4: 0.225, s=5: 0.269, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.294; cross heading r = 0.271.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.261 [0.227, 0.313] | -0.0168 [-0.0307, -0.00346] |
| 1 | 0.249 [0.209, 0.315] | -0.0248 [-0.0413, -0.00693] |
| 2 | 0.242 [0.198, 0.319] | -0.0239 [-0.0436, 0.00351] |
| 3 | 0.247 [0.195, 0.369] | -0.0265 [-0.0521, 0.0167] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.247 [0.195, 0.369]

| n bin | entities | R_real | R_pred | 1/n | real/pred |
|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | n/a [n/a, n/a] |
| 2 | 172 | 0.716 | 0.624 | 0.5 | n/a [n/a, n/a] |
| 3-4 | 197 | 0.538 | 0.464 | 0.288 | n/a [n/a, n/a] |
| 5-8 | 146 | 0.458 | 0.361 | 0.152 | 1.87 [1.37, 3.16] |
| 9-16 | 236 | 0.37 | 0.313 | 0.0882 | 1.09 [0.788, 1.29] |
| 17+ | 15 | 0.352 | 0.29 | 0.0564 | 1.21 |

## Residual parallax on textured land (A15)

p90_px_scale1: 7.82, p90_px_scale2: 3.91, p90_px_scale4: 1.96, p90_px_scale8: 0.978, n_pixels: 1.95e+05

Runtime 0.4 min.
