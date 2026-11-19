# Stage A summary: matrixcity block3, source cache (metric3d_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 1.78 [1.19, 2.55], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 6.61, beta = 0, last change 0.000515 (converged).
Block datum: calibration (factor 1.0814).
Median relative depth residual: label-free with parallax datum 0.159, after datum 0.157, oracle per-frame fit 0.113.

## Records (A7)

100687 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.931 m; additivity max relative difference 8.09e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 14.3 [8.19, 22.6] | 19.9 | 0.0514 | 0.0965 | 0.462 | 596 | 583 |
| land | 628 | 5.41 [3.89, 7.57] | 11 | 0.0924 | 0.174 | 0.431 | 184 | 173 |
| water | 501 | 26.9 [20.2, 37.5] | 27.2 | 0 | 0 | 0.5 | 1.11e+03 | 1.1e+03 |
| roof | 162 | 5.85 [3.42, 13.5] | 11.4 | 0.0617 | 0.0988 | 0.456 | 200 | 187 |
| ground | 466 | 5.3 [3.89, 7.48] | 10.8 | 0.103 | 0.2 | 0.423 | 178 | 168 |
| contrast_q1 | 134 | 5.08 [2.65, 8.56] | 7.2 | 0.112 | 0.224 | 0.429 | 82 | 70.4 |
| contrast_q2 | 139 | 4.42 [3.38, 7.48] | 7.93 | 0.0791 | 0.173 | 0.434 | 97.8 | 86.6 |
| contrast_q3 | 163 | 5.75 [3.91, 7.74] | 13.5 | 0.0859 | 0.117 | 0.444 | 274 | 263 |
| contrast_q4 | 107 | 6.75 [5.78, 9.22] | 11.8 | 0.112 | 0.168 | 0.42 | 211 | 200 |
| contrast_q5 | 85 | 4.7 [3.66, 7.35] | 13.6 | 0.0706 | 0.212 | 0.421 | 279 | 269 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 87 | 4.38 | 4.15 [2.61, 5.53] | 0.244 |
| 3 | 87 | 4.42 | 4.15 [2.61, 5.53] | 0.244 |
| 5 | 87 | 4.28 | 4.15 [2.61, 5.53] | 0.244 |
| 9 | 87 | 4.28 | 4.15 [2.61, 5.53] | 0.244 |

## Repetition (A10)

r(s), same strip: s=1: 0.889, s=2: 0.825, s=3: 0.702, s=4: 0.499, s=5: 0.265, s=6: 0.152, s=7: 0.0332, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.802; cross heading r = 0.711.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.481 [0.4, 0.603] | 0.307 [0.17, 0.373] |
| 1 | 0.413 [0.331, 0.546] | 0.263 [0.0965, 0.328] |
| 2 | 0.33 [0.25, 0.494] | 0.211 [0.0272, 0.29] |
| 3 | 0.237 [0.162, 0.441] | 0.157 [-0.086, 0.246] |
| 5 | 0.0925 [-0.00471, 0.301] | 0.0848 [-0.305, 0.223] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.237 [0.162, 0.441]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.927 | 0.619 | 0.5 | 1.5 [1.3, 1.6] | 1.46 |
| 3-4 | 228 | 0.775 | 0.454 | 0.284 | 1.71 [1.04, 2.26] | 1.68 |
| 5-8 | 230 | 0.863 | 0.355 | 0.155 | 2.43 [1.73, 3.02] | 2.33 |
| 9-16 | 275 | 0.774 | 0.3 | 0.0823 | 2.58 [1.6, 3.54] | 2.44 |
| 17+ | 624 | 0.892 | 0.269 | 0.0412 | 3.32 [2, 4.68] | 3.5 |

## Residual parallax on textured land (A15)

p90_px_scale1: 110, p90_px_scale2: 55, p90_px_scale4: 27.5, p90_px_scale8: 13.8, n_pixels: 3.5e+05

Runtime 8.8 min.
