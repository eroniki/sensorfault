# Stage A summary: matrixcity block3, source cache (metric3d_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **refuted**.
- A8 single-frame calibration on held-out land: s_z = 1.79 [1.2, 2.55], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 6.61, beta = 0, last change 0.000515 (converged).
Block datum: calibration (factor 1.0814).
Median relative depth residual: label-free with parallax datum 0.159, after datum 0.157, oracle per-frame fit 0.113.

## Records (A7)

100690 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.931 m; additivity max relative difference 6.45e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 14.4 [8.17, 22.8] | 20 | 0.0469 | 0.0965 | 0.462 | 604 | 591 |
| land | 628 | 5.46 [3.9, 7.7] | 11.3 | 0.0844 | 0.174 | 0.432 | 193 | 181 |
| water | 501 | 27.1 [20.4, 37.6] | 27.3 | 0 | 0 | 0.5 | 1.12e+03 | 1.11e+03 |
| roof | 162 | 6.01 [3.38, 13.7] | 11.6 | 0.0494 | 0.105 | 0.455 | 206 | 193 |
| ground | 466 | 5.36 [3.89, 7.51] | 11.1 | 0.0966 | 0.197 | 0.424 | 188 | 177 |
| contrast_q1 | 134 | 5.1 [2.64, 8.52] | 7.27 | 0.104 | 0.224 | 0.429 | 83.4 | 71.9 |
| contrast_q2 | 139 | 4.4 [3.33, 7.51] | 8.02 | 0.0719 | 0.18 | 0.433 | 99.9 | 88.8 |
| contrast_q3 | 163 | 5.77 [3.89, 7.96] | 13.8 | 0.0675 | 0.117 | 0.445 | 289 | 277 |
| contrast_q4 | 107 | 6.84 [5.78, 9.71] | 12.1 | 0.112 | 0.159 | 0.419 | 221 | 211 |
| contrast_q5 | 85 | 4.84 [3.72, 7.4] | 14 | 0.0706 | 0.212 | 0.424 | 296 | 286 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 87 | 4.46 | 4.24 [2.6, 5.56] | 0.246 |
| 3 | 87 | 4.46 | 4.24 [2.6, 5.56] | 0.246 |
| 5 | 87 | 4.45 | 4.24 [2.6, 5.56] | 0.246 |
| 9 | 87 | 4.45 | 4.24 [2.6, 5.56] | 0.246 |

## Repetition (A10)

r(s), same strip: s=1: 0.889, s=2: 0.825, s=3: 0.702, s=4: 0.498, s=5: 0.265, s=6: 0.152, s=7: 0.0333, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.802; cross heading r = 0.712.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.481 [0.399, 0.603] | 0.304 [0.169, 0.369] |
| 1 | 0.413 [0.331, 0.547] | 0.26 [0.0961, 0.324] |
| 2 | 0.33 [0.249, 0.494] | 0.207 [0.0262, 0.285] |
| 3 | 0.237 [0.161, 0.44] | 0.152 [-0.0866, 0.239] |
| 5 | 0.0925 [-0.00495, 0.301] | 0.0745 [-0.306, 0.21] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.237 [0.161, 0.44]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.926 | 0.618 | 0.5 | 1.5 [1.3, 1.6] | 1.45 |
| 3-4 | 228 | 0.773 | 0.454 | 0.284 | 1.7 [1.04, 2.25] | 1.68 |
| 5-8 | 230 | 0.863 | 0.355 | 0.155 | 2.43 [1.73, 3.02] | 2.34 |
| 9-16 | 275 | 0.772 | 0.3 | 0.0823 | 2.58 [1.59, 3.55] | 2.44 |
| 17+ | 624 | 0.893 | 0.268 | 0.0412 | 3.33 [2, 4.69] | 3.51 |

## Residual parallax on textured land (A15)

p90_px_scale1: 110, p90_px_scale2: 55, p90_px_scale4: 27.5, p90_px_scale8: 13.8, n_pixels: 3.5e+05

Runtime 9.1 min.
