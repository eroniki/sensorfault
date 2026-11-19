# Stage A summary: matrixcity block3, source cache (unidepth_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.37 [0.944, 2.06], FAIL (noise model at fault, H1 undefined).

## Alignment (A5)

Mode metric, gauge k = 1.112, beta = 0, last change 0.000309 (converged).
Block datum: calibration (factor 1.0142).
Median relative depth residual: label-free with parallax datum 0.0139, after datum 0.0064, oracle per-frame fit 0.00638.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.283 m; additivity max relative difference 8.2e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 4.5 [2.83, 7.65] | 10.8 | 0.132 | 0.206 | 0.401 | 169 | 165 |
| land | 628 | 3.97 [2.44, 5.72] | 12.6 | 0.167 | 0.242 | 0.374 | 231 | 228 |
| water | 501 | 5.04 [2.51, 12.3] | 7.93 | 0.0878 | 0.162 | 0.435 | 90.6 | 86.7 |
| roof | 162 | 7.12 [5.01, 9.31] | 17.9 | 0.0123 | 0.0309 | 0.489 | 478 | 471 |
| ground | 466 | 3.06 [1.66, 4.49] | 10.1 | 0.221 | 0.315 | 0.334 | 146 | 143 |
| contrast_q1 | 134 | 4.01 [2.01, 5.34] | 9.56 | 0.164 | 0.209 | 0.387 | 134 | 129 |
| contrast_q2 | 139 | 3.86 [2.54, 5.4] | 10.2 | 0.151 | 0.223 | 0.38 | 152 | 148 |
| contrast_q3 | 163 | 3.59 [1.82, 7] | 14 | 0.19 | 0.27 | 0.353 | 288 | 284 |
| contrast_q4 | 107 | 3.75 [2.44, 8.33] | 15.3 | 0.159 | 0.234 | 0.381 | 343 | 339 |
| contrast_q5 | 85 | 4.56 [1.61, 7.35] | 13.5 | 0.165 | 0.282 | 0.372 | 267 | 265 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 88 | 6.41 | 2.39 [1.64, 3.23] | 0.403 |
| 3 | 90 | 6.18 | 2.31 [1.4, 3.22] | 0.429 |
| 5 | 90 | 6.2 | 2.31 [1.4, 3.22] | 0.429 |
| 9 | 90 | 6.22 | 2.31 [1.4, 3.22] | 0.429 |

## Repetition (A10)

r(s), same strip: s=1: 0.441, s=2: 0.286, s=3: 0.192, s=4: 0.142, s=5: 0.181, s=6: 0.194, s=7: 0.153, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.189; cross heading r = 0.218.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.227 [0.102, 0.387] | 0.0778 [0.0459, 0.148] |
| 1 | 0.191 [0.056, 0.378] | 0.0664 [0.0364, 0.14] |
| 2 | 0.172 [-0.0254, 0.386] | 0.0556 [0.028, 0.124] |
| 3 | 0.168 [-0.0976, 0.442] | 0.0484 [0.0246, 0.127] |
| 5 | 0.173 [-0.37, 0.647] | 0.0195 [0.00973, 0.105] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.168 [-0.0976, 0.442]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.586 | 0.584 | 0.5 | 1 [0.833, 1.43] | 2.56 |
| 3-4 | 228 | 0.441 | 0.404 | 0.284 | 1.09 [0.755, 1.83] | 0.97 |
| 5-8 | 230 | 0.259 | 0.296 | 0.155 | 0.876 [0.391, 1.85] | 1.15 |
| 9-16 | 275 | 0.315 | 0.236 | 0.0823 | 1.33 [0.705, 4.4] | 1.94 |
| 17+ | 624 | 0.316 | 0.202 | 0.0412 | 1.57 [0.67, 8.37] | 1.38 |

## Residual parallax on textured land (A15)

p90_px_scale1: 10.7, p90_px_scale2: 5.37, p90_px_scale4: 2.69, p90_px_scale8: 1.34, n_pixels: 3.5e+05

Runtime 8.2 min.
