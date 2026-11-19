# Stage A summary: synthetic block3, source exact_repeated, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **supported**.
- A8 single-frame calibration on held-out land: s_z = 1.01 [0.991, 1.04], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9993, beta = 0, last change 0.000337 (converged).
Block datum: calibration (factor 1.0007).
Median relative depth residual: label-free with parallax datum 0.00248, after datum 0.00241, oracle per-frame fit 0.00241.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.312 m; additivity max relative difference 4.77e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.95 [1.85, 2.05] | 2.04 | 0.244 | 0.443 | 0.321 | 1.05 | 2.62 |
| land | 274 | 1.96 [1.88, 2.13] | 2.11 | 0.248 | 0.431 | 0.323 | 1.36 | 2.95 |
| water | 123 | 1.95 [1.57, 2.07] | 1.88 | 0.236 | 0.472 | 0.317 | 0.357 | 1.91 |
| roof | 36 | 1.87 | 2.12 | 0.278 | 0.472 | 0.326 | 1.75 | 3 |
| ground | 238 | 1.99 [1.88, 2.17] | 2.11 | 0.244 | 0.424 | 0.322 | 1.3 | 2.94 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 5 | 0.88 | 1.97 [1.86, 2.12] | 0.438 |
| 3 | 4 | 0.891 | 1.97 [1.86, 2.12] | 0.438 |
| 5 | 4 | 0.886 | 1.97 [1.86, 2.12] | 0.438 |
| 9 | 4 | 0.886 | 1.97 [1.86, 2.12] | 0.438 |

## Repetition (A10)

r(s), same strip: s=1: 0.229, s=2: 0.216, s=3: 0.216, s=4: 0.214, s=5: 0.219, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.235; cross heading r = 0.216.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.219 [0.194, 0.267] | 0.00404 [-0.00673, 0.0191] |
| 1 | 0.216 [0.189, 0.284] | 0.00725 [-0.00579, 0.026] |
| 2 | 0.216 [0.188, 0.289] | 0.019 [0.00276, 0.0464] |
| 3 | 0.216 [0.176, 0.333] | 0.0247 [0.00524, 0.0548] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.216 [0.176, 0.333]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 172 | 0.605 | 0.608 | 0.5 | 0.995 [0.881, 1.09] | 0.993 |
| 3-4 | 197 | 0.441 | 0.442 | 0.288 | 0.996 [0.822, 1.16] | 1.04 |
| 5-8 | 146 | 0.36 | 0.335 | 0.152 | 1.07 [0.869, 1.27] | 1.06 |
| 9-16 | 236 | 0.302 | 0.286 | 0.0882 | 1.06 [0.793, 1.23] | 1.13 |
| 17+ | 15 | 0.289 | 0.261 | 0.0564 | 1.11 | 1.13 |

## Residual parallax on textured land (A15)

p90_px_scale1: 2.17, p90_px_scale2: 1.09, p90_px_scale4: 0.543, p90_px_scale8: 0.272, n_pixels: 1.95e+05

Runtime 0.5 min.
