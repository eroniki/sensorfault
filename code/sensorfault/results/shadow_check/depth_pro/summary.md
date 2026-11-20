# Shadow check: MatrixCity block3, depth_pro, step 4

185 frames. Alignment: Stage A label-free pair graph + calibration datum (factor 0.98133; converged True); per-frame scale differs from the oracle (median exact/raw on calibration tiles) by median 0.024, p90 0.095.

Error variants: `dh_lf` label-free aligned; `dh_or` oracle scale; `dh_loc` label-free minus the frame's flat-ground median of its 64 m tile (removes the per-frame scale error). Height errors in metres, + = raised. CIs: 95% tile bootstrap over 64 m tiles.

## Part 1, shadow mask win31_thr0.6

Median shadow share of flat ground per frame 0.00991; centers {'shadow': 54266, 'edge': 500251, 'none': 546167}.

### (a) Height-error step across edges (dark minus bright side, 5 px)

| variant | group | n | mean abs step | mean signed step | median contrast | mean abs, >10 px from real steps |
|---|---|---|---|---|---|---|
| dh_lf | shadow_masksides | 45739 | 0.401 [0.377, 0.422] | -0.022 [-0.0668, 0.0158] | 0.531 | 0.359 [0.314, 0.395] |
| dh_lf | shadow | 53262 | 0.377 [0.357, 0.398] | -0.0589 [-0.114, -0.0117] | 0.557 | 0.362 [0.325, 0.391] |
| dh_lf | edge | 499271 | 0.318 [0.301, 0.337] | -0.0145 [-0.0403, 0.00766] | 0.531 | 0.3 [0.279, 0.322] |
| dh_lf | none | 544524 | 0.259 [0.242, 0.28] | 0.000149 [-0.0144, 0.0114] | 0.209 | 0.25 [0.227, 0.279] |
| dh_lf | edge_contrast_matched | 499271 | 0.316 [0.3, 0.334] | -0.0154 [-0.0405, 0.00645] | n/a |  |
| dh_or | shadow_masksides | 45739 | 0.399 [0.377, 0.419] | -0.0222 [-0.0674, 0.0162] | 0.531 |  |
| dh_or | shadow | 53262 | 0.376 [0.357, 0.394] | -0.0593 [-0.115, -0.0123] | 0.557 |  |
| dh_or | edge | 499271 | 0.317 [0.3, 0.336] | -0.0151 [-0.041, 0.00669] | 0.531 |  |
| dh_or | none | 544524 | 0.259 [0.242, 0.278] | -0.000436 [-0.0148, 0.0109] | 0.209 |  |
| dh_or | edge_contrast_matched | 499271 | 0.315 [0.299, 0.334] | -0.0159 [-0.042, 0.00625] | n/a |  |
| dh_loc | shadow_masksides | 45656 | 0.529 [0.488, 0.575] | 0.0189 [-0.0356, 0.073] | 0.532 |  |
| dh_loc | shadow | 53152 | 0.508 [0.469, 0.561] | -0.0195 [-0.0831, 0.0393] | 0.557 |  |
| dh_loc | edge | 499215 | 0.463 [0.424, 0.511] | 0.0339 [-0.0123, 0.1] | 0.531 |  |
| dh_loc | none | 544316 | 0.393 [0.363, 0.429] | 0.0215 [-0.0109, 0.0508] | 0.209 |  |
| dh_loc | edge_contrast_matched | 499215 | 0.463 [0.424, 0.511] | 0.0365 [-0.0117, 0.104] | n/a |  |

### (b) Shadow minus lit on the same 8 m entity

| variant | entities | entity-frames | mean (m) | median (m) | share lowered |
|---|---|---|---|---|---|
| dh_lf | 693 | 4264 | 0.0253 [-0.0407, 0.0921] | 0.0465 | 0.47 |
| dh_or | 693 | 4264 | 0.0235 [-0.0423, 0.088] | 0.0437 | 0.469 |
| dh_loc | 691 | 4255 | 0.0235 [-0.0427, 0.0902] | 0.0442 | 0.472 |

### (c) Spearman: entity mean |dh| vs boundary-band share

| variant | entities | rho_s [CI] |
|---|---|---|
| dh_lf | 2312 | 0.0798 [-0.0962, 0.218] |
| dh_or | 2312 | 0.0265 [-0.108, 0.138] |
| dh_loc | 2300 | 0.0147 [-0.0715, 0.106] |

### Repetition on 1 m cells (>= 5 frames)

| variant | cells | n cells | median frames | rms (m) | rho all pairs | null | rho same heading | null |
|---|---|---|---|---|---|---|---|---|
| dh_lf | band | 7555 | 17 | 16.1 | 0.588 [0.426, 0.695] | 0.418 [0.238, 0.532] | 0.839 [0.67, 0.942] | 0.58 [0.392, 0.693] |
| dh_lf | other | 81845 | 23 | 14.5 | 0.366 [0.26, 0.478] | 0.177 [0.104, 0.252] | 0.622 [0.505, 0.744] | 0.322 [0.228, 0.408] |
| dh_lf | all | 89400 | 23 | 14.6 | 0.379 [0.279, 0.486] | 0.193 [0.118, 0.272] | 0.637 [0.525, 0.753] | 0.342 [0.25, 0.427] |
| dh_loc | band | 7523 | 17 | 3.42 | 0.328 [0.275, 0.398] | 0.0205 [0.0116, 0.033] | 0.505 [0.437, 0.586] | 0.023 [0.0135, 0.0378] |
| dh_loc | other | 81701 | 23 | 3.56 | 0.363 [0.297, 0.42] | 0.00932 [0.00768, 0.0109] | 0.631 [0.552, 0.707] | 0.0114 [0.00873, 0.0142] |
| dh_loc | all | 89224 | 23 | 3.55 | 0.361 [0.3, 0.419] | 0.00989 [0.00839, 0.0114] | 0.624 [0.548, 0.701] | 0.0121 [0.00965, 0.0148] |
| dh_or | band | 7555 | 17 | 20.8 | 0.107 [0.039, 0.245] | 0.0821 [0.0317, 0.18] | 0.169 [0.0745, 0.364] | 0.115 [0.0495, 0.247] |
| dh_or | other | 81845 | 23 | 17.4 | 0.124 [0.0753, 0.191] | 0.0309 [0.0215, 0.0454] | 0.197 [0.131, 0.292] | 0.088 [0.0585, 0.126] |
| dh_or | all | 89400 | 23 | 17.6 | 0.122 [0.0719, 0.196] | 0.0347 [0.0239, 0.0513] | 0.195 [0.13, 0.29] | 0.0902 [0.0599, 0.129] |

## Part 1, shadow mask win61_thr0.6

Median shadow share of flat ground per frame 0.0423; centers {'shadow': 141439, 'edge': 520011, 'none': 545361}.

### (a) Height-error step across edges (dark minus bright side, 5 px)

| variant | group | n | mean abs step | mean signed step | median contrast | mean abs, >10 px from real steps |
|---|---|---|---|---|---|---|
| dh_lf | shadow_masksides | 129110 | 0.393 [0.375, 0.411] | -0.0401 [-0.0741, -0.0123] | 0.486 | 0.365 [0.341, 0.389] |
| dh_lf | shadow | 140067 | 0.358 [0.339, 0.373] | -0.081 [-0.117, -0.0454] | 0.441 | 0.346 [0.326, 0.37] |
| dh_lf | edge | 518926 | 0.304 [0.288, 0.326] | -0.00506 [-0.0239, 0.0132] | 0.483 | 0.287 [0.264, 0.306] |
| dh_lf | none | 543503 | 0.256 [0.239, 0.275] | 0.00245 [-0.0103, 0.0126] | 0.206 | 0.248 [0.221, 0.287] |
| dh_lf | edge_contrast_matched | 518926 | 0.306 [0.289, 0.327] | -0.00359 [-0.0217, 0.0147] | n/a |  |
| dh_or | shadow_masksides | 129110 | 0.392 [0.375, 0.412] | -0.0399 [-0.0733, -0.0122] | 0.486 |  |
| dh_or | shadow | 140067 | 0.357 [0.34, 0.372] | -0.081 [-0.119, -0.0442] | 0.441 |  |
| dh_or | edge | 518926 | 0.304 [0.287, 0.326] | -0.00564 [-0.0247, 0.0118] | 0.483 |  |
| dh_or | none | 543503 | 0.255 [0.239, 0.271] | 0.00186 [-0.0115, 0.0121] | 0.206 |  |
| dh_or | edge_contrast_matched | 518926 | 0.305 [0.289, 0.327] | -0.00414 [-0.0225, 0.0136] | n/a |  |
| dh_loc | shadow_masksides | 129018 | 0.569 [0.528, 0.636] | 0.000287 [-0.0497, 0.048] | 0.486 |  |
| dh_loc | shadow | 139927 | 0.527 [0.475, 0.579] | -0.0137 [-0.0593, 0.0346] | 0.441 |  |
| dh_loc | edge | 518807 | 0.456 [0.418, 0.507] | 0.0414 [-0.00477, 0.11] | 0.483 |  |
| dh_loc | none | 543319 | 0.391 [0.363, 0.424] | 0.0317 [0.00277, 0.0558] | 0.206 |  |
| dh_loc | edge_contrast_matched | 518807 | 0.458 [0.42, 0.509] | 0.0437 [-0.00468, 0.115] | n/a |  |

### (b) Shadow minus lit on the same 8 m entity

| variant | entities | entity-frames | mean (m) | median (m) | share lowered |
|---|---|---|---|---|---|
| dh_lf | 970 | 7916 | -0.04 [-0.092, 0.00915] | -0.0279 | 0.538 |
| dh_or | 970 | 7916 | -0.0453 [-0.107, 0.00593] | -0.0284 | 0.537 |
| dh_loc | 966 | 7904 | -0.0391 [-0.091, 0.0102] | -0.0272 | 0.537 |

### (c) Spearman: entity mean |dh| vs boundary-band share

| variant | entities | rho_s [CI] |
|---|---|---|
| dh_lf | 2312 | 0.078 [-0.0969, 0.233] |
| dh_or | 2312 | 0.0469 [-0.091, 0.15] |
| dh_loc | 2300 | -0.015 [-0.0994, 0.0607] |

### Repetition on 1 m cells (>= 5 frames)

| variant | cells | n cells | median frames | rms (m) | rho all pairs | null | rho same heading | null |
|---|---|---|---|---|---|---|---|---|
| dh_lf | band | 15292 | 19 | 16 | 0.593 [0.444, 0.689] | 0.407 [0.262, 0.509] | 0.829 [0.684, 0.947] | 0.564 [0.412, 0.676] |
| dh_lf | other | 74108 | 24 | 14.4 | 0.346 [0.244, 0.455] | 0.157 [0.0918, 0.234] | 0.6 [0.483, 0.719] | 0.3 [0.224, 0.382] |
| dh_lf | all | 89400 | 23 | 14.6 | 0.379 [0.279, 0.477] | 0.193 [0.119, 0.263] | 0.637 [0.525, 0.75] | 0.345 [0.254, 0.432] |
| dh_loc | band | 15253 | 19 | 3.4 | 0.327 [0.273, 0.405] | 0.0286 [0.0181, 0.0372] | 0.488 [0.421, 0.585] | 0.0301 [0.0194, 0.0406] |
| dh_loc | other | 73971 | 24 | 3.58 | 0.365 [0.29, 0.421] | 0.00874 [0.00708, 0.0105] | 0.644 [0.564, 0.724] | 0.0112 [0.00815, 0.0145] |
| dh_loc | all | 89224 | 23 | 3.55 | 0.361 [0.303, 0.426] | 0.011 [0.00901, 0.0125] | 0.624 [0.551, 0.723] | 0.0137 [0.0107, 0.0165] |
| dh_or | band | 15292 | 19 | 20 | 0.115 [0.0408, 0.265] | 0.082 [0.0396, 0.164] | 0.173 [0.072, 0.382] | 0.114 [0.0592, 0.224] |
| dh_or | other | 74108 | 24 | 17.2 | 0.124 [0.0711, 0.194] | 0.0276 [0.0189, 0.0399] | 0.2 [0.129, 0.287] | 0.0887 [0.0602, 0.129] |
| dh_or | all | 89400 | 23 | 17.6 | 0.122 [0.0694, 0.208] | 0.036 [0.0258, 0.0563] | 0.195 [0.122, 0.294] | 0.0932 [0.0625, 0.137] |

## Part 1, shadow mask win31_thr0.5

Median shadow share of flat ground per frame 0.00237; centers {'shadow': 18900, 'edge': 485391, 'none': 549383}.

### (a) Height-error step across edges (dark minus bright side, 5 px)

| variant | group | n | mean abs step | mean signed step | median contrast | mean abs, >10 px from real steps |
|---|---|---|---|---|---|---|
| dh_lf | shadow_masksides | 14106 | 0.377 [0.346, 0.406] | -0.0107 [-0.0667, 0.0253] | 0.591 | 0.305 [0.234, 0.361] |
| dh_lf | shadow | 18234 | 0.371 [0.339, 0.403] | -0.0239 [-0.0996, 0.0325] | 0.619 | 0.357 [0.285, 0.415] |
| dh_lf | edge | 484275 | 0.335 [0.311, 0.362] | -0.0272 [-0.0587, 0.00348] | 0.596 | 0.313 [0.292, 0.334] |
| dh_lf | none | 547544 | 0.264 [0.247, 0.283] | -0.0015 [-0.0182, 0.0103] | 0.212 | 0.252 [0.228, 0.279] |
| dh_lf | edge_contrast_matched | 484275 | 0.332 [0.308, 0.356] | -0.0311 [-0.0639, -0.00196] | n/a |  |
| dh_or | shadow_masksides | 14106 | 0.374 [0.345, 0.405] | -0.00982 [-0.0661, 0.0263] | 0.591 |  |
| dh_or | shadow | 18234 | 0.369 [0.339, 0.399] | -0.024 [-0.1, 0.0321] | 0.619 |  |
| dh_or | edge | 484275 | 0.332 [0.309, 0.357] | -0.0276 [-0.059, 0.00254] | 0.596 |  |
| dh_or | none | 547544 | 0.263 [0.247, 0.281] | -0.00217 [-0.0179, 0.00945] | 0.212 |  |
| dh_or | edge_contrast_matched | 484275 | 0.329 [0.306, 0.353] | -0.0315 [-0.0642, -0.00276] | n/a |  |
| dh_loc | shadow_masksides | 14061 | 0.47 [0.432, 0.525] | 0.0268 [-0.0422, 0.0809] | 0.592 |  |
| dh_loc | shadow | 18158 | 0.475 [0.438, 0.509] | -0.0238 [-0.107, 0.0585] | 0.619 |  |
| dh_loc | edge | 484197 | 0.474 [0.434, 0.526] | 0.0145 [-0.0341, 0.07] | 0.596 |  |
| dh_loc | none | 547322 | 0.4 [0.373, 0.438] | 0.0257 [0.000217, 0.05] | 0.212 |  |
| dh_loc | edge_contrast_matched | 484197 | 0.472 [0.431, 0.522] | 0.013 [-0.0363, 0.0668] | n/a |  |

### (b) Shadow minus lit on the same 8 m entity

| variant | entities | entity-frames | mean (m) | median (m) | share lowered |
|---|---|---|---|---|---|
| dh_lf | 377 | 1672 | 0.0295 [-0.0778, 0.138] | 0.0103 | 0.485 |
| dh_or | 377 | 1672 | 0.0283 [-0.0772, 0.136] | 0.0123 | 0.485 |
| dh_loc | 373 | 1666 | 0.0273 [-0.0849, 0.139] | 0.00951 | 0.488 |

### (c) Spearman: entity mean |dh| vs boundary-band share

| variant | entities | rho_s [CI] |
|---|---|---|
| dh_lf | 2312 | 0.037 [-0.115, 0.171] |
| dh_or | 2312 | 0.00324 [-0.133, 0.102] |
| dh_loc | 2300 | -0.00598 [-0.0844, 0.0759] |

### Repetition on 1 m cells (>= 5 frames)

| variant | cells | n cells | median frames | rms (m) | rho all pairs | null | rho same heading | null |
|---|---|---|---|---|---|---|---|---|
| dh_lf | band | 3215 | 15 | 16.5 | 0.631 [0.478, 0.73] | 0.477 [0.298, 0.577] | 0.887 [0.736, 0.966] | 0.653 [0.47, 0.748] |
| dh_lf | other | 86185 | 23 | 14.6 | 0.374 [0.271, 0.48] | 0.184 [0.111, 0.266] | 0.63 [0.521, 0.742] | 0.329 [0.247, 0.408] |
| dh_lf | all | 89400 | 23 | 14.6 | 0.379 [0.279, 0.477] | 0.192 [0.118, 0.261] | 0.637 [0.525, 0.75] | 0.339 [0.247, 0.422] |
| dh_loc | band | 3196 | 15 | 3.53 | 0.341 [0.272, 0.426] | 0.0416 [0.0227, 0.0619] | 0.51 [0.408, 0.619] | 0.0566 [0.0362, 0.0862] |
| dh_loc | other | 86028 | 23 | 3.56 | 0.362 [0.293, 0.413] | 0.00949 [0.00776, 0.0111] | 0.627 [0.552, 0.711] | 0.0111 [0.00849, 0.0137] |
| dh_loc | all | 89224 | 23 | 3.55 | 0.361 [0.303, 0.426] | 0.0101 [0.00827, 0.0116] | 0.624 [0.551, 0.723] | 0.0121 [0.00922, 0.0146] |
| dh_or | band | 3215 | 15 | 20.9 | 0.132 [0.0456, 0.306] | 0.108 [0.0396, 0.246] | 0.194 [0.0735, 0.443] | 0.145 [0.0559, 0.313] |
| dh_or | other | 86185 | 23 | 17.5 | 0.122 [0.0683, 0.194] | 0.0319 [0.0218, 0.0481] | 0.195 [0.126, 0.289] | 0.0875 [0.0597, 0.127] |
| dh_or | all | 89400 | 23 | 17.6 | 0.122 [0.0694, 0.208] | 0.034 [0.024, 0.0541] | 0.195 [0.122, 0.294] | 0.0893 [0.0598, 0.129] |

## Part 2: plane sweep (full resolution, same-heading pairs)

Selection: {'faulty_pool': 19, 'faulty_selected': 19, 'control_pool': 997, 'control_selected': 19, 'seconds': 75.21904134750366}

| group | entities | median abs nu_depth | median abs nu_photo | share photo closer | share gradient toward truth | share abs nu_photo <= 0.3 m | median stiffness | Spearman(stiffness, texture) |
|---|---|---|---|---|---|---|---|---|
| faulty | 19 | 18.8 | 0.3 | 1 | 0.632 | 0.737 | 0.568 | 0.307 |
| control | 19 | 5.91 | 0.1 | 1 | 0.789 | 0.895 | 1.1 | 0.43 |

nu_depth outside the +-6 m sweep and the local variant (nu_depth_loc = shadow-region mean of dh_loc, i.e. the frame's 64 m tile median removed; the gradient is taken at the clipped value):

- faulty: outside sweep 19/19; inside sweep {'entities': 0, 'share_photo_closer': None, 'share_gradient_toward_truth': None}; local {'median_abs_nu_depth_loc': 1.1767185329735237, 'outside_sweep': 1, 'share_photo_closer': 0.8421052631578947, 'share_gradient_toward_truth': 0.8888888888888888, 'abs_ge_0.5_entities': 15, 'abs_ge_0.5_share_photo_closer': 0.9333333333333333, 'abs_ge_0.5_share_gradient_toward_truth': 0.8666666666666667}; stiff/soft halves: closer n/a / n/a
- control: outside sweep 9/19; inside sweep {'entities': 10, 'share_photo_closer': 1.0, 'share_gradient_toward_truth': 0.9}; local {'median_abs_nu_depth_loc': 1.1264783203468915, 'outside_sweep': 1, 'share_photo_closer': 1.0, 'share_gradient_toward_truth': 1.0, 'abs_ge_0.5_entities': 14, 'abs_ge_0.5_share_photo_closer': 1.0, 'abs_ge_0.5_share_gradient_toward_truth': 1.0}; stiff/soft halves: closer n/a / n/a

Figures: panels_part1_steps.png, panels_part2.png. Runtime 15.2 min.
