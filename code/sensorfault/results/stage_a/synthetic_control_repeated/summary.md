# Stage A summary: synthetic block3, source exact_repeated, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.01 [0.992, 1.04], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9993, beta = 0, last change 0.000337 (converged).
Block datum: calibration (factor 1.0007).
Median relative depth residual: label-free with parallax datum 0.00248, after datum 0.00241, oracle per-frame fit 0.00241.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.312 m; additivity max relative difference 3.84e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.95 [1.84, 2.06] | 2.05 | 0.244 | 0.438 | 0.322 | 1.11 | 2.67 |
| land | 274 | 1.95 [1.88, 2.13] | 2.13 | 0.248 | 0.427 | 0.324 | 1.44 | 3.01 |
| water | 123 | 1.93 [1.58, 2.06] | 1.88 | 0.236 | 0.463 | 0.317 | 0.371 | 1.92 |
| roof | 36 | 1.87 | 2.12 | 0.278 | 0.444 | 0.325 | 1.75 | 2.99 |
| ground | 238 | 2.01 [1.88, 2.18] | 2.13 | 0.244 | 0.424 | 0.324 | 1.39 | 3.02 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 5 | 0.882 | 1.99 [1.87, 2.12] | 0.434 |
| 3 | 4 | 0.887 | 1.99 [1.87, 2.12] | 0.434 |
| 5 | 4 | 0.886 | 1.99 [1.87, 2.12] | 0.434 |
| 9 | 4 | 0.886 | 1.99 [1.87, 2.12] | 0.434 |

## Repetition (A10)

r(s), same strip: s=1: 0.229, s=2: 0.217, s=3: 0.216, s=4: 0.214, s=5: 0.219, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.235; cross heading r = 0.216.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.219 [0.194, 0.267] | 0.00427 [-0.00649, 0.0193] |
| 1 | 0.217 [0.189, 0.284] | 0.00753 [-0.00556, 0.0262] |
| 2 | 0.217 [0.188, 0.289] | 0.0194 [0.00306, 0.0468] |
| 3 | 0.217 [0.176, 0.333] | 0.0251 [0.00564, 0.0554] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.217 [0.176, 0.333]

| n bin | entities | R_real | R_pred | 1/n | real/pred |
|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | n/a [n/a, n/a] |
| 2 | 172 | 0.604 | 0.608 | 0.5 | n/a [n/a, n/a] |
| 3-4 | 197 | 0.458 | 0.443 | 0.288 | n/a [n/a, n/a] |
| 5-8 | 146 | 0.356 | 0.335 | 0.152 | 1.99 [1.24, 2.47] |
| 9-16 | 236 | 0.322 | 0.286 | 0.0882 | 1.02 [0.675, 1.28] |
| 17+ | 15 | 0.293 | 0.261 | 0.0564 | 1.12 |

## Residual parallax on textured land (A15)

p90_px_scale1: 2.17, p90_px_scale2: 1.09, p90_px_scale4: 0.543, p90_px_scale8: 0.272, n_pixels: 1.95e+05

Runtime 0.5 min.
