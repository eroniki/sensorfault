# Stage A summary: synthetic block3, source exact_independent, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 0.998 [0.973, 1.02], pass.

## Alignment (A5)

Mode metric, gauge k = 0.9961, beta = 0, last change 0.000126 (converged).
Block datum: calibration (factor 1.0039).
Median relative depth residual: label-free with parallax datum 0.00391, after datum 0.00135, oracle per-frame fit 0.00135.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.313 m; additivity max relative difference 4.43e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.03 [0.98, 1.09] | 1.02 | 0.64 | 0.942 | 0.0215 | -6.93 | 0.00119 |
| land | 274 | 1.03 [0.947, 1.12] | 1.02 | 0.65 | 0.938 | 0.0211 | -7.04 | 0.000909 |
| water | 123 | 1.03 [0.978, 1.11] | 1.03 | 0.618 | 0.951 | 0.0265 | -6.67 | 0.00193 |
| roof | 36 | 1.13 | 1.03 | 0.611 | 0.972 | 0.0646 | -6.65 | 0.00359 |
| ground | 238 | 1.02 [0.947, 1.12] | 1.01 | 0.655 | 0.933 | 0.0153 | -7.1 | 0.000655 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 3 | 1.02 | 1 [0.934, 1.12] | 0.941 |
| 3 | 2 | 1.01 | 1.02 [0.947, 1.12] | 0.945 |
| 5 | 2 | 1.01 | 1.02 [0.947, 1.12] | 0.945 |
| 9 | 2 | 1.01 | 1.02 [0.947, 1.12] | 0.945 |

## Repetition (A10)

r(s), same strip: s=1: 0.0215, s=2: -0.0301, s=3: 0.0162, s=4: 0.0197, s=5: 0.016, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00734; cross heading r = 0.00594.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.00866 [-0.0117, 0.0521] | -0.0124 [-0.0434, 0.00607] |
| 1 | 0.00545 [-0.0158, 0.0615] | -0.0138 [-0.0559, 0.0114] |
| 2 | 0.0173 [-0.0167, 0.107] | -0.0138 [-0.0682, 0.0209] |
| 3 | 0.0179 [-0.0393, 0.0927] | -0.0237 [-0.096, 0.0153] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.0179 [-0.0393, 0.0927]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 172 | 0.492 | 0.509 | 0.5 | 0.966 [0.846, 1.09] | 0.692 |
| 3-4 | 197 | 0.261 | 0.301 | 0.288 | 0.868 [0.664, 1.06] | 0.705 |
| 5-8 | 146 | 0.163 | 0.167 | 0.152 | 0.978 [0.681, 1.3] | 0.7 |
| 9-16 | 236 | 0.0987 | 0.105 | 0.0882 | 0.945 [0.7, 1.16] | 0.676 |
| 17+ | 15 | 0.0404 | 0.0733 | 0.0564 | 0.551 | 0.454 |

## Residual parallax on textured land (A15)

p90_px_scale1: 1.11, p90_px_scale2: 0.553, p90_px_scale4: 0.277, p90_px_scale8: 0.138, n_pixels: 1.95e+05

Runtime 0.5 min.
