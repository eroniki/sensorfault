# Stage A summary: matrixcity block3, source cache (depth_anything_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.26 [1, 1.51], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode inverse, gauge k = 1.136e+04, beta = 0, last change 0.4 (not below 0.1%).
Block datum: calibration (factor 3.9184).
Median relative depth residual: label-free with parallax datum 0.776, after datum 0.366, oracle per-frame fit 0.329.

## Records (A7)

100683 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.19 m; additivity max relative difference 6.75e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 3.98 [2.84, 5.6] | 7.23 | 0.172 | 0.234 | 0.375 | 87.2 | 71.1 |
| land | 628 | 4.11 [3.1, 5.68] | 8.87 | 0.153 | 0.213 | 0.388 | 125 | 110 |
| water | 501 | 3.79 [2.06, 6.29] | 4.4 | 0.196 | 0.259 | 0.359 | 39.4 | 23.1 |
| roof | 162 | 6.09 [1.78, 7.61] | 12 | 0.173 | 0.204 | 0.365 | 224 | 207 |
| ground | 466 | 3.8 [2.8, 5.26] | 7.48 | 0.146 | 0.217 | 0.396 | 91.2 | 76.4 |
| contrast_q1 | 134 | 3.2 [2.5, 5.36] | 4.91 | 0.209 | 0.246 | 0.352 | 45.3 | 30 |
| contrast_q2 | 139 | 5.2 [3.9, 6.02] | 7.65 | 0.151 | 0.187 | 0.377 | 96.1 | 80.1 |
| contrast_q3 | 163 | 4.52 [3.11, 6.93] | 11.6 | 0.129 | 0.196 | 0.409 | 210 | 195 |
| contrast_q4 | 107 | 3.74 [2.44, 5.35] | 9.19 | 0.121 | 0.215 | 0.418 | 133 | 118 |
| contrast_q5 | 85 | 4 [2.53, 5.25] | 9 | 0.153 | 0.235 | 0.383 | 127 | 113 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 84 | 1.59 | 3.4 [2.12, 4.53] | 0.308 |
| 3 | 85 | 1.65 | 3.23 [2.21, 4.39] | 0.298 |
| 5 | 85 | 1.65 | 3.23 [2.21, 4.39] | 0.298 |
| 9 | 85 | 1.65 | 3.23 [2.21, 4.39] | 0.298 |

## Repetition (A10)

r(s), same strip: s=1: 0.944, s=2: 0.953, s=3: 0.989, s=4: 1.01, s=5: 1.19, s=6: 1.27, s=7: 1.32, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.454; cross heading r = 0.0271.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 1.1 [0.786, 1.51] | 0.425 [0.22, 0.482] |
| 1 | 1.12 [0.737, 1.6] | 0.42 [0.189, 0.483] |
| 2 | 1.16 [0.643, 1.75] | 0.411 [0.143, 0.486] |
| 3 | 1.2 [0.607, 1.98] | 0.401 [0.07, 0.496] |
| 5 | 1.29 [0.112, 2.37] | 0.384 [-0.103, 0.494] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 1.2 [0.607, 1.98]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.99 | 1 | 0.5 | 0.99 [0.944, 1.21] | 0.968 |
| 3-4 | 228 | 0.924 | 1 | 0.284 | 0.924 [0.841, 1.28] | 0.855 |
| 5-8 | 230 | 0.783 | 1 | 0.155 | 0.783 [0.419, 1.24] | 0.785 |
| 9-16 | 275 | 0.779 | 1 | 0.0823 | 0.779 [0.487, 1.21] | 0.683 |
| 17+ | 624 | 0.523 | 1 | 0.0412 | 0.523 [0.34, 0.916] | 0.215 |

## Residual parallax on textured land (A15)

p90_px_scale1: 467, p90_px_scale2: 233, p90_px_scale4: 117, p90_px_scale8: 58.3, n_pixels: 3.5e+05

Runtime 8.9 min.
