# Stage A summary: synthetic block3, source exact_repeated, step 20

Frames 54 in 6 strips; entities 3904 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 843, 'held_out': 998, 'buffer': 2063}.
Synthetic block.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **supported**.
- A8 single-frame calibration on held-out land: s_z = 1.02 [0.988, 1.07], pass.
- Noise model: white pixel noise sigma (global 0.001899) plus a record floor tau (global 0.002589, per-frame median 0.002589, p10-p90 0.002421-0.002785).
- A8 s_z by pixels per record (held-out land): 4-9 px 1.05, 10-12 px 0.979, 13-16 px 1.04.

## Alignment (A5)

Mode metric, gauge k = 0.9993, beta = 0, last change 0.000337 (converged).
Block datum: calibration (factor 1.0007).
Median relative depth residual: label-free with parallax datum 0.00248, after datum 0.00241, oracle per-frame fit 0.00241.

## Records (A7)

23660 records, 3402 entities seen, median n = 4, median pixels per record 9, median shape remainder 0.312 m; additivity max relative difference 3.79e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 397 | 1.8 [1.74, 1.91] | 1.94 | 0.242 | 0.504 | 0.32 | 0.323 | 2.14 |
| land | 274 | 1.83 [1.72, 1.99] | 2.04 | 0.23 | 0.5 | 0.326 | 0.77 | 2.58 |
| water | 123 | 1.77 [1.62, 1.9] | 1.69 | 0.268 | 0.512 | 0.307 | -0.672 | 1.22 |
| roof | 36 | 1.92 | 2.03 | 0.194 | 0.444 | 0.352 | 0.81 | 2.55 |
| ground | 238 | 1.79 [1.72, 2.08] | 2.04 | 0.235 | 0.508 | 0.322 | 0.764 | 2.59 |
| contrast_q1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q2 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q3 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q4 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| contrast_q5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 3 | 0.785 | 1.83 [1.72, 1.98] | 0.5 |
| 3 | 5 | 0.785 | 1.77 [1.72, 1.91] | 0.513 |
| 5 | 4 | 0.786 | 1.83 [1.72, 1.98] | 0.5 |
| 9 | 4 | 0.786 | 1.83 [1.72, 1.98] | 0.5 |

## Repetition (A10)

r(s), same strip: s=1: 0.229, s=2: 0.216, s=3: 0.216, s=4: 0.214, s=5: 0.219, s=6: n/a, s=7: n/a, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.235; cross heading r = 0.216.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.222 [0.2, 0.248] | -0.00774 [-0.02, 0.00147] |
| 1 | 0.216 [0.195, 0.247] | -0.00687 [-0.0273, 0.0115] |
| 2 | 0.216 [0.199, 0.235] | 0.0166 [-0.00488, 0.049] |
| 3 | 0.215 [0.189, 0.246] | 0.0322 [0.0188, 0.0595] |
| 5 | n/a [n/a, n/a] | n/a [n/a, n/a] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.215 [0.189, 0.246]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 105 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 172 | 0.605 | 0.607 | 0.5 | 0.996 [0.896, 1.07] | 0.943 |
| 3-4 | 197 | 0.441 | 0.441 | 0.288 | 0.999 [0.868, 1.14] | 0.977 |
| 5-8 | 146 | 0.36 | 0.334 | 0.152 | 1.08 [0.937, 1.22] | 1.05 |
| 9-16 | 236 | 0.302 | 0.284 | 0.0882 | 1.06 [0.922, 1.15] | 1.05 |
| 17+ | 15 | 0.289 | 0.259 | 0.0564 | 1.12 | 1.11 |

## Residual parallax on textured land (A15)

p90_px_scale1: 2.17, p90_px_scale2: 1.09, p90_px_scale4: 0.543, p90_px_scale8: 0.272, n_pixels: 1.95e+05

Runtime 0.6 min.
