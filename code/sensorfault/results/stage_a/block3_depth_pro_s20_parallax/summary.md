# Stage A summary: matrixcity block3, source cache (depth_pro), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.09 [0.759, 1.34], pass.
- Noise model (class): white pixel noise sigma (global 0.0008394) plus a record floor tau (global 0.01839, per-frame median 0.01839, p10-p90 0.008752-0.04754); floor shape log g: roof +1.255, steep +0.562, log_pixels -0.054, log_incidence -0.161, log_depth +2.908 (g p10-p90 0.637-1.82).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.09, 6-8 px 1.03, 9-184 px 1.26; by class: roof 0.595, ground 1.21.

## Alignment (A5)

Mode metric, gauge k = 75.55, beta = 0, last change 0.000769 (converged).
Block datum: parallax (factor 1).
Median relative depth residual: label-free with parallax datum 0.0495, after datum 0.0495, oracle per-frame fit 0.0338.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.157 m; additivity max relative difference 5.73e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 6.8 [3.65, 9.71] | 17.9 | 0.105 | 0.181 | 0.416 | 483 | 473 |
| land | 628 | 3.41 [2.03, 5.42] | 18.9 | 0.182 | 0.304 | 0.354 | 534 | 524 |
| water | 501 | 9.85 [6.76, 25.6] | 16.7 | 0.00798 | 0.0259 | 0.495 | 418 | 408 |
| roof | 162 | 1.9 [1.11, 4.59] | 8.01 | 0.253 | 0.463 | 0.287 | 98.4 | 88.6 |
| ground | 466 | 3.95 [2.43, 6.27] | 21.4 | 0.157 | 0.249 | 0.377 | 686 | 676 |
| contrast_q1 | 134 | 3.39 [2.11, 7.3] | 7.26 | 0.149 | 0.276 | 0.374 | 80 | 71.7 |
| contrast_q2 | 139 | 3.39 [1.7, 6.17] | 8.2 | 0.173 | 0.353 | 0.355 | 101 | 92.9 |
| contrast_q3 | 163 | 3.99 [1.81, 5.93] | 28.6 | 0.178 | 0.27 | 0.357 | 1.23e+03 | 1.22e+03 |
| contrast_q4 | 107 | 2.75 [1.82, 4.61] | 13.3 | 0.215 | 0.327 | 0.341 | 263 | 255 |
| contrast_q5 | 85 | 2.82 [2.05, 5.18] | 25.4 | 0.212 | 0.306 | 0.33 | 968 | 959 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 87 | 3.06 | 2.08 [1.43, 4.11] | 0.442 |
| 3 | 84 | 3.35 | 2.11 [1.43, 4.24] | 0.446 |
| 5 | 84 | 3.37 | 2.11 [1.43, 4.24] | 0.446 |
| 9 | 84 | 3.37 | 2.11 [1.43, 4.24] | 0.446 |

## Repetition (A10)

r(s), same strip: s=1: 0.93, s=2: 0.862, s=3: 0.797, s=4: 0.813, s=5: 0.978, s=6: 0.869, s=7: 0.454, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.848; cross heading r = 0.847.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.88 [0.785, 0.988] | 0.725 [0.613, 0.814] |
| 1 | 0.842 [0.727, 0.983] | 0.691 [0.591, 0.797] |
| 2 | 0.822 [0.672, 1.05] | 0.641 [0.543, 0.756] |
| 3 | 0.858 [0.624, 1.23] | 0.611 [0.488, 0.784] |
| 5 | 0.845 [0.262, 1.86] | 0.484 [0.00881, 0.804] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.858 [0.624, 1.23]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.878 | 0.929 | 0.5 | 0.945 [0.811, 1.11] | 0.885 |
| 3-4 | 228 | 0.785 | 0.898 | 0.284 | 0.873 [0.73, 1.1] | 0.747 |
| 5-8 | 230 | 0.67 | 0.88 | 0.155 | 0.761 [0.518, 1.04] | 0.602 |
| 9-16 | 275 | 0.779 | 0.87 | 0.0823 | 0.896 [0.641, 1.2] | 0.951 |
| 17+ | 624 | 0.885 | 0.864 | 0.0412 | 1.02 [0.788, 1.42] | 0.999 |

## Residual parallax on textured land (A15)

p90_px_scale1: 34, p90_px_scale2: 17, p90_px_scale4: 8.49, p90_px_scale8: 4.24, n_pixels: 3.5e+05

Runtime 7.1 min.
