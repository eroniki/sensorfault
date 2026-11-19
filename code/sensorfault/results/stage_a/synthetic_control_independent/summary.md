# Stage A summary: synthetic block3, source exact_independent, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **refuted**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 0.993 [0.966, 1.02], pass.
- Noise model: white pixel noise sigma (global 0.001879) plus a record floor tau (global 0.0003063, per-frame median 0.0003063, p10-p90 2.611e-05-0.0004613).
- A8 s_z by pixels per record (held-out land): 4-9 px 1.01, 10-12 px 0.977, 13-16 px 0.91.

## Alignment (A5)

Mode metric, gauge k = 0.9961, beta = 0, last change 0.000126 (converged).
Block datum: calibration (factor 1.0039).
Median relative depth residual: label-free with parallax datum 0.00391, after datum 0.00135, oracle per-frame fit 0.00135.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.313 m; additivity max relative difference 4.94e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.03 [0.982, 1.06] | 1.01 | 0.642 | 0.945 | 0.0143 | -6.92 | 0.000269 |
| land | 274 | 1.02 [0.967, 1.09] | 1.01 | 0.646 | 0.938 | 0.0168 | -7.03 | 0.000284 |
| water | 123 | 1.04 [0.996, 1.06] | 1.01 | 0.634 | 0.959 | 0.025 | -6.69 | 0.000237 |
| roof | 36 | 1.06 | 1.03 | 0.583 | 0.944 | 0.0456 | -6.63 | 0.00305 |
| ground | 238 | 1.01 [0.965, 1.09] | 1.01 | 0.655 | 0.937 | 0.013 | -7.08 | 0.00012 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 2 | 1.01 | 1 [0.927, 1.12] | 0.93 |
| 3 | 1 | 1.01 | 1.02 [0.967, 1.09] | 0.938 |
| 5 | 1 | 1.01 | 1.02 [0.967, 1.09] | 0.938 |
| 9 | 1 | 1.01 | 1.02 [0.967, 1.09] | 0.938 |

## Repetition (A10)

r(s), same strip: s=1: 0.0207, s=2: -0.0298, s=3: 0.0166, s=4: 0.0201, s=5: 0.0159, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.00754; cross heading r = 0.00568.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.00515 [-0.00936, 0.0339] | -0.00877 [-0.0166, -0.000903] |
| 1 | -0.00731 [-0.0307, 0.0315] | -0.0106 [-0.0284, 0.0115] |
| 2 | 0.0175 [-0.0157, 0.135] | -0.00675 [-0.0386, 0.0315] |
| 3 | 0.0192 [-0.0176, 0.18] | -0.0286 [-0.0625, 0.0223] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.0192 [-0.0176, 0.18]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 172 | 0.492 | 0.51 | 0.5 | 0.965 [0.811, 1.11] | 0.686 |
| 3-4 | 197 | 0.261 | 0.302 | 0.288 | 0.864 [0.622, 1.07] | 0.698 |
| 5-8 | 146 | 0.163 | 0.168 | 0.152 | 0.971 [0.607, 1.25] | 0.682 |
| 9-16 | 236 | 0.0985 | 0.106 | 0.0882 | 0.932 [0.519, 1.15] | 0.667 |
| 17+ | 15 | 0.0404 | 0.0746 | 0.0564 | 0.541 | 0.422 |

## Residual parallax on textured land (A15)

p90_px_scale1: 1.11, p90_px_scale2: 0.553, p90_px_scale4: 0.277, p90_px_scale8: 0.138, n_pixels: 1.95e+05

Runtime 0.5 min.
