# E1 summary: MatrixCity small_city aerial train block_3 (float32 depth)

Views: 185; surfels: 81152; runtime 9092.6 s.

## Clean scenes

| noise scale | temperature | per-view FA (%) | mean T/m | loop: seeds with any exclusion / seeds | untestable views (%) |
|---|---|---|---|---|---|
| 1.0 | on | 7.5 | 0.965 | 0 / 20 | 0.0 |
| 1.0 | off | 2.8 | 0.965 | 0 / 20 | 0.0 |
| 1.3 | on | 7.7 | 1.631 | 0 / 20 | 0.0 |
| 1.3 | off | 100.0 | 1.631 | 20 / 20 | 0.0 |

## One faulty view: detection at matched 5% per-view false alarm (single pass)

Gate / aligned RMS detection (%) per cell; rows: magnitude (sigma), columns: area fraction.

| delta \ area | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|
| 2 | 100.0 / 70.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| 4 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| 8 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |
| 16 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 |

Outer loop with Holm across rounds: faulty view excluded (%) and clean views excluded per scene.

| delta \ area | 0.05 | 0.1 | 0.2 | 0.4 |
|---|---|---|---|---|
| 2 | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) |
| 4 | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) |
| 8 | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.00) |
| 16 | 100.0 (0.00) | 100.0 (0.00) | 100.0 (0.20) | 100.0 (0.00) |

## Same error on three neighbouring frames of a strip (8 sigma, 25 m patch)

| noise scale | B | temperature | loop | whitened | faulty views detected (%) | clean views excluded per scene | Holm alarm (cap) |
|---|---|---|---|---|---|---|---|
| 1.0 | 1 | on | off | yes | 93.3 | 13.30 | 10 |
| 1.0 | 1 | on | on | yes | 93.3 | 0.00 | 0 |
| 1.0 | 3 | off | on | yes | 93.3 | 0.00 | 0 |
| 1.0 | 3 | on | off | yes | 96.7 | 15.10 | 10 |
| 1.0 | 3 | on | on | no | 93.3 | 2.30 | 0 |
| 1.0 | 3 | on | on | yes | 93.3 | 0.00 | 0 |
| 1.3 | 1 | on | off | yes | 93.3 | 9.00 | 10 |
| 1.3 | 1 | on | on | yes | 93.3 | 0.00 | 0 |
| 1.3 | 3 | off | on | yes | 93.3 | 7.20 | 10 |
| 1.3 | 3 | on | off | yes | 93.3 | 11.10 | 10 |
| 1.3 | 3 | on | on | no | 93.3 | 2.60 | 0 |
| 1.3 | 3 | on | on | yes | 93.3 | 0.00 | 0 |

## Against the predictions (only what this run can address)

- Holm family-wise false alarm, clean scenes, nominal noise: 0/20 seeds with any exclusion (alpha = 0.05).
- Scene temperature with noise 1.3x the offline level (H6, 'no temperature raises clean FA'): per-view FA 7.7% with temperature vs 100.0% without; loop exclusions in 0/20 vs 20/20 seeds.
- H5 ('raw T/m > 1.2 from independent blocks'): raw mean T/m = 0.965 here. Not addressed: this run has one surfel per pixel and independent noise, the case in which the blocks are exact; H5 concerns real, correlated depth error.
- H3 (gate vs cheap baselines, >= 15 points): only the aligned-RMS baseline is run, on injected regional faults rather than specular masks. Mean gain over cells with delta >= 4: 0.0 points; overall 100.0% vs 98.1% at matched 5% per-view FA. Partial evidence only.
- H6 leave-block-out (noise 1.0x): B=3 detects 93.3% vs B=1 93.3% of repeated faulty views (prediction: >= 10 points) -> not supported in this run.
- H6 outer loop (noise 1.0x, B=3): loop 93.3% vs single pass 96.7% (prediction: >= 10 points) -> not supported in this run.
- H6 leave-block-out (noise 1.3x): B=3 detects 93.3% vs B=1 93.3% of repeated faulty views (prediction: >= 10 points) -> not supported in this run.
- H6 outer loop (noise 1.3x, B=3): loop 93.3% vs single pass 93.3% (prediction: >= 10 points) -> not supported in this run.
- Not addressed by this run: H1, H2 (need a real depth model), H4 (needs training), H7 (pose check), and the remaining H6 ablations.
