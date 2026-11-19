# Stage A summary: matrixcity block3, source cache (unidepth_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **supported**.
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.05 [0.919, 1.24], pass.
- Noise model (class): white pixel noise sigma (global 0.001517) plus a record floor tau (global 0.003442, per-frame median 0.003442, p10-p90 0.002299-0.005733); floor shape log g: roof +0.538, steep +0.472, log_pixels -0.061, log_incidence +0.017, log_depth -1.799 (g p10-p90 0.683-1.91).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.09, 6-8 px 1.02, 9-184 px 1.07; by class: roof 1.08, ground 1.05.

## Alignment (A5)

Mode metric, gauge k = 1.112, beta = 0, last change 0.000309 (converged).
Block datum: calibration (factor 1.0142).
Median relative depth residual: label-free with parallax datum 0.0139, after datum 0.0064, oracle per-frame fit 0.00638.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.283 m; additivity max relative difference 5.37e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 3.66 [2.75, 5.77] | 6.34 | 0.131 | 0.227 | 0.399 | 55.5 | 53.2 |
| land | 628 | 2.67 [1.83, 3.19] | 3.73 | 0.21 | 0.352 | 0.34 | 16.3 | 15.4 |
| water | 501 | 7.11 [3.25, 12.3] | 8.55 | 0.0319 | 0.0699 | 0.472 | 105 | 102 |
| roof | 162 | 3.22 [2.3, 3.85] | 4.57 | 0.111 | 0.204 | 0.409 | 29 | 25.2 |
| ground | 466 | 2.5 [1.57, 3.16] | 3.38 | 0.245 | 0.403 | 0.317 | 11.9 | 12 |
| contrast_q1 | 134 | 2.92 [1.89, 3.55] | 3.43 | 0.187 | 0.313 | 0.36 | 14 | 12.5 |
| contrast_q2 | 139 | 2.54 [1.75, 3.05] | 3.47 | 0.194 | 0.367 | 0.354 | 14 | 12.8 |
| contrast_q3 | 163 | 2.53 [1.82, 3.32] | 3.89 | 0.209 | 0.337 | 0.342 | 17.9 | 17.2 |
| contrast_q4 | 107 | 2.68 [1.66, 3.7] | 3.56 | 0.224 | 0.374 | 0.325 | 14 | 13.7 |
| contrast_q5 | 85 | 2.89 [1.33, 4.17] | 4.39 | 0.259 | 0.388 | 0.304 | 23.3 | 22.9 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 77 | 2.1 | 2.27 [1.64, 2.9] | 0.4 |
| 3 | 77 | 2.13 | 2.21 [1.51, 2.84] | 0.413 |
| 5 | 79 | 2.1 | 2.21 [1.51, 2.83] | 0.41 |
| 9 | 79 | 2.1 | 2.21 [1.51, 2.83] | 0.41 |

## Repetition (A10)

r(s), same strip: s=1: 0.441, s=2: 0.286, s=3: 0.192, s=4: 0.142, s=5: 0.181, s=6: 0.194, s=7: 0.153, s=8: n/a, s=9: n/a, s=10: n/a
Cross strip r = 0.189; cross heading r = 0.218.

| B | rho_hat(B) | permutation null |
|---|---|---|
| 0 | 0.322 [0.23, 0.45] | 0.12 [0.066, 0.2] |
| 1 | 0.231 [0.153, 0.369] | 0.0996 [0.0507, 0.175] |
| 2 | 0.177 [0.107, 0.322] | 0.0788 [0.033, 0.144] |
| 3 | 0.156 [0.0828, 0.331] | 0.0706 [0.0266, 0.14] |
| 5 | 0.191 [0.105, 0.608] | 0.0603 [0.00255, 0.122] |
| 9 | n/a [n/a, n/a] | n/a [n/a, n/a] |

## H2: the law (A11), rho_hat(B=3) = 0.156 [0.0828, 0.331]

Fused error for the law: equal-weight mean of the frames' errors. The last column is real/pred for the information-weighted map, for reference.

| n bin | entities | R_real | R_pred | 1/n | real/pred | info-weighted |
|---|---|---|---|---|---|---|
| 1 | 108 | 1 | 1 | 1 | 1 [1, 1] | 1 |
| 2 | 95 | 0.586 | 0.578 | 0.5 | 1.01 [0.799, 1.53] | 0.695 |
| 3-4 | 228 | 0.441 | 0.396 | 0.284 | 1.11 [0.803, 1.73] | 1.14 |
| 5-8 | 230 | 0.259 | 0.286 | 0.155 | 0.906 [0.46, 1.73] | 1.29 |
| 9-16 | 275 | 0.315 | 0.225 | 0.0823 | 1.4 [0.723, 2.63] | 1.15 |
| 17+ | 624 | 0.316 | 0.191 | 0.0412 | 1.66 [0.822, 2.94] | 1.52 |

## Residual parallax on textured land (A15)

p90_px_scale1: 10.7, p90_px_scale2: 5.37, p90_px_scale4: 2.69, p90_px_scale8: 1.34, n_pixels: 3.5e+05

Runtime 6.5 min.
