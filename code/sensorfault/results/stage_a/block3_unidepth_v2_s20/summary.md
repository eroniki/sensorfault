# Stage A summary: matrixcity block3, source cache (unidepth_v2), step 20

Frames 185 in 27 strips; entities 8382 (side 8.0 m; split tiles 64.0 m, groups of 2): {'calibration': 1767, 'held_out': 1995, 'buffer': 4620}.
Tables use one block only and are marked **same scene**.

## Decisions

- **H1** (lower bound of s_z on held-out land, n >= 5, above 1.25): **undefined (A8 failed)** (without the A8 gate: supported).
- **H2** (realised/predicted fused variance in [0.67, 1.5] per n-bin): **inconclusive**.
- A8 single-frame calibration on held-out land: s_z = 1.29 [0.956, 1.8], FAIL (noise model at fault, H1 undefined).
- Noise model: white pixel noise sigma (global 0.001517) plus a record floor tau (global 0.003257, per-frame median 0.003257, p10-p90 0.00211-0.005383).
- A8 s_z by pixels per record (held-out land): 4-5 px 1.19, 6-8 px 1.16, 9-184 px 1.77.

## Alignment (A5)

Mode metric, gauge k = 1.112, beta = 0, last change 0.000309 (converged).
Block datum: calibration (factor 1.0142).
Median relative depth residual: label-free with parallax datum 0.0139, after datum 0.0064, oracle per-frame fit 0.00638.

## Records (A7)

100691 records, 6639 entities seen, median n = 12, median pixels per record 6, median shape remainder 0.283 m; additivity max relative difference 6.78e-16.

## H1: depth-only map on held-out tiles, n >= 5 (A9)

| group | entities | s_z | s_z tail | cov 0.683 | cov 0.95 | ECE | log-score | calib. loss |
|---|---|---|---|---|---|---|---|---|
| all | 1129 | 4.11 [2.67, 6.16] | 8.4 | 0.133 | 0.229 | 0.399 | 101 | 97.9 |
| land | 628 | 3.4 [2.08, 4.64] | 9.35 | 0.178 | 0.293 | 0.364 | 125 | 123 |
| water | 501 | 5.02 [2.49, 10.4] | 7.03 | 0.0758 | 0.148 | 0.443 | 69.7 | 66.8 |
| roof | 162 | 5.62 [4.15, 6.89] | 12.1 | 0.0247 | 0.0802 | 0.476 | 216 | 211 |
| ground | 466 | 2.68 [1.54, 3.89] | 8.16 | 0.232 | 0.367 | 0.326 | 93.9 | 92.2 |
| contrast_q1 | 134 | 3.56 [2.11, 4.91] | 6.95 | 0.142 | 0.239 | 0.372 | 68.1 | 65 |
| contrast_q2 | 139 | 3.36 [1.93, 4.93] | 7.51 | 0.194 | 0.309 | 0.363 | 79.4 | 77 |
| contrast_q3 | 163 | 3.29 [1.74, 4.91] | 10.2 | 0.19 | 0.313 | 0.358 | 151 | 149 |
| contrast_q4 | 107 | 3.08 [1.99, 5.29] | 12.5 | 0.187 | 0.29 | 0.38 | 228 | 225 |
| contrast_q5 | 85 | 3.62 [1.41, 5.73] | 8.93 | 0.176 | 0.318 | 0.347 | 113 | 112 |

Consensus gate (A12), land, after removing flagged frames:

| B | frames removed | temperature | s_z | cov 0.95 |
|---|---|---|---|---|
| 1 | 86 | 4.54 | 2.03 [1.37, 3] | 0.451 |
| 3 | 84 | 4.66 | 2.05 [1.33, 3] | 0.451 |
| 5 | 88 | 4.52 | 2.27 [1.34, 3.11] | 0.436 |
| 9 | 88 | 4.52 | 2.27 [1.34, 3.11] | 0.436 |

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
| 2 | 95 | 0.586 | 0.578 | 0.5 | 1.01 [0.799, 1.53] | 0.874 |
| 3-4 | 228 | 0.441 | 0.396 | 0.284 | 1.11 [0.803, 1.73] | 1.26 |
| 5-8 | 230 | 0.259 | 0.286 | 0.155 | 0.906 [0.46, 1.73] | 0.999 |
| 9-16 | 275 | 0.315 | 0.225 | 0.0823 | 1.4 [0.723, 2.63] | 1.26 |
| 17+ | 624 | 0.316 | 0.191 | 0.0412 | 1.66 [0.822, 2.94] | 1.37 |

## Residual parallax on textured land (A15)

p90_px_scale1: 10.7, p90_px_scale2: 5.37, p90_px_scale4: 2.69, p90_px_scale8: 1.34, n_pixels: 3.5e+05

Runtime 6.5 min.
