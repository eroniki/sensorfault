# Stage A summary: matrixcity block3, source cache (depth_anything_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.26 [1.01, 1.51], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode inverse, gauge k = 1.136e+04, beta = 0, last change 0.4 (not below 0.1%).
Block datum: calibration (factor 3.9184).
Median relative depth residual: label-free with parallax datum 0.776, after datum 0.366, oracle per-frame fit 0.329.

## Records (A7)

100683 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.19 m; additivity max relative difference 7.58e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 4 [2.86, 5.48] | 7.21 | 0.168 | 0.23 | 0.375 | 86.6 | 70.5 |
| land | 628 | 4.15 [3.15, 5.59] | 8.82 | 0.15 | 0.207 | 0.388 | 124 | 109 |
| water | 501 | 3.81 [2.08, 6.29] | 4.42 | 0.192 | 0.259 | 0.359 | 39.6 | 23.3 |
| roof | 162 | 6.09 [1.78, 7.62] | 12 | 0.173 | 0.204 | 0.365 | 223 | 206 |
| ground | 466 | 3.84 [2.82, 5.34] | 7.42 | 0.142 | 0.208 | 0.396 | 89.7 | 75 |
| contrast_q1 | 134 | 3.21 [2.48, 5.43] | 4.92 | 0.187 | 0.246 | 0.354 | 45.4 | 30.1 |
| contrast_q2 | 139 | 5.2 [3.92, 6] | 7.58 | 0.151 | 0.187 | 0.378 | 94.5 | 78.6 |
| contrast_q3 | 163 | 4.51 [2.98, 6.88] | 11.6 | 0.129 | 0.202 | 0.408 | 208 | 192 |
| contrast_q4 | 107 | 3.89 [2.51, 5.32] | 9.17 | 0.121 | 0.178 | 0.418 | 133 | 118 |
| contrast_q5 | 85 | 4.12 [2.52, 5.15] | 8.91 | 0.165 | 0.224 | 0.384 | 124 | 111 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 89 | 1.55 | 3.33 [2.24, 4.45] | 0.284 |
| 3 | 84 | 1.64 | 3.27 [2.25, 4.37] | 0.291 |
| 5 | 88 | 1.57 | 3.32 [2.21, 4.39] | 0.281 |
| 9 | 88 | 1.57 | 3.32 [2.21, 4.39] | 0.281 |

## Repetition (A10)

r(s), same strip: s=1: 0.916, s=2: 0.924, s=3: 0.959, s=4: 0.979, s=5: 1.15, s=6: 1.22, s=7: 1.28, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.44; cross heading r = 0.0262.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 1.06 [0.76, 1.46] | 0.415 [0.22, 0.468] |
| 1 | 1.09 [0.732, 1.55] | 0.41 [0.187, 0.468] |
| 2 | 1.12 [0.641, 1.7] | 0.402 [0.142, 0.47] |
| 3 | 1.16 [0.599, 1.94] | 0.394 [0.0694, 0.48] |
| 5 | 1.25 [0.112, 2.27] | 0.38 [-0.103, 0.487] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 1.16 [0.599, 1.94]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.99 | 1 | 0.5 | 0.99 [0.947, 1.22] | 0.969 |
| 3-4 | 228 | 0.915 | 1 | 0.284 | 0.915 [0.799, 1.27] | 0.852 |
| 5-8 | 230 | 0.784 | 1 | 0.155 | 0.784 [0.42, 1.24] | 0.785 |
| 9-16 | 275 | 0.779 | 1 | 0.0823 | 0.779 [0.487, 1.21] | 0.684 |
| 17+ | 624 | 0.503 | 1 | 0.0412 | 0.503 [0.336, 0.908] | 0.206 |

## Residual parallax on textured land (A15)

p90_px_scale1: 467, p90_px_scale2: 233, p90_px_scale4: 117, p90_px_scale8: 58.3, n_pixels: 3.5e+05

Runtime 8.7 min.
