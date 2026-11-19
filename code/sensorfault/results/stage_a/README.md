# Stage A results

`experiments/stage_a_honesty.py` outputs, one folder per run: `summary.md`, `summary.json`,
`strips.md`, `classes.md` and `law_figure.json`. `records.npz` is not committed.

## MatrixCity block 3

Common settings for every block 3 run:

- **Code:** commit `fc5d728`, run 2026-11-19 on 3101 (RTX 2070, 16 cores). Seed 0. Noise model `--noise class`, the default: white pixel noise plus a record floor whose size depends on surface class, incidence, depth and pixels per record. Default flags apart from those in the table.
- **Data:** `transforms_block3.json` (185 frames) with exact depth from `block_3_depth`.
- **RGB:** the `block_3` images, used for the contrast classes.
- **Depth cache:** monocular depth from `/home/eiki/data/depth_cache`, stored as float16. Metric3D was cached with `max_depth_m=1000`. The environment is recorded in `/home/eiki/data/depth_cache/ENV.txt`.

| folder | source | model | step | datum | noise | alignment |
|---|---|---|---|---|---|---|
| `block3_depth_pro_s20` | cache | Depth Pro | 20 | calibration | class (covariate floor) | converged |
| `block3_depth_pro_s20_parallax` | cache | Depth Pro | 20 | parallax | class (covariate floor) | converged |
| `block3_depth_pro_s10` | cache | Depth Pro | 10 | calibration | class (covariate floor) | converged |
| `block3_metric3d_v2_s20` | cache | Metric3D v2 ViT-L | 20 | calibration | class (covariate floor) | converged |
| `block3_unidepth_v2_s20` | cache | UniDepthV2 ViT-L | 20 | calibration | class (covariate floor) | converged |
| `block3_depth_anything_v2_s20` | cache | Depth Anything V2 Large (inverse, scale only) | 20 | calibration | class (covariate floor) | **alignment not converged (2-cycle)** |
| `block3_control_independent` | exact_independent | — | 20 | calibration | white 0.04445, planted rho 0, class (covariate floor) | converged |
| `block3_control_repeated` | exact_repeated | — | 20 | calibration | white 0.002, planted tau 2.0, rho 0.3, class (covariate floor) | **alignment not converged (2-cycle)** |

### Notes

- **Not converged.** Two runs cycle between two states in the label-free pair-graph rounds:
  - **exact_repeated:** one frame cycles, frame 56 (yaw 270, end of strip 26). It swings by about 0.20 in log scale each round.
  - **DAv2:** most frames cycle, by 0.2–0.3 in log scale. The scale-only inverse model probably does not fit DAv2's affine-invariant disparity.
- **A8.** Single-frame calibration on held-out land; it passes when s_z is in [0.9, 1.1]. Where A8 fails, H1 is undefined. The summaries also give the decision without the A8 gate. Held-out land s_z:
  - **Pass:**
    - Depth Pro s20, parallax datum: 1.09 [0.76, 1.34]
    - Depth Pro s10: 1.10 [0.86, 1.29]
    - UniDepthV2: 1.05 [0.92, 1.24]
    - independent control: 1.00 [0.92, 1.07]
    - repeated control: 1.05 [1.00, 1.09]
  - **Fail:**
    - Depth Pro s20: 1.16 [0.81, 1.44]
    - Metric3D v2: 1.20 [0.97, 1.43]
    - DAv2: 1.18 [0.91, 1.45]
- **Two folders sharing a name.** `synthetic_control_*` are the synthetic-block controls, not block 3 results. They also use the covariate record-floor noise model.
