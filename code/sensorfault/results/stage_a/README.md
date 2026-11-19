# Stage A results

`experiments/stage_a_honesty.py` outputs, one folder per run: `summary.md`, `summary.json`,
`strips.md`, `classes.md` and `law_figure.json`. `records.npz` is not committed.

## MatrixCity block 3

Common settings for every block 3 run:

- **Code:** commit `b886adb` (record weights from the neighbourhood median depth), run 2026-11-19 on 3101 (RTX 2070, 16 cores). Seed 0. Default flags apart from those in the table.
- **Data:** `transforms_block3.json` (185 frames) with exact depth from `block_3_depth`.
- **RGB:** the `block_3` images, used for the contrast classes.
- **Depth cache:** monocular depth from `/home/eiki/data/depth_cache`, stored as float16. Metric3D was cached with `max_depth_m=1000`. The environment is recorded in `/home/eiki/data/depth_cache/ENV.txt`.

| folder | source | model | step | datum | noise | alignment |
|---|---|---|---|---|---|---|
| `block3_depth_pro_s20` | cache | Depth Pro | 20 | calibration | — | converged |
| `block3_depth_pro_s20_parallax` | cache | Depth Pro | 20 | parallax | — | converged |
| `block3_depth_pro_s10` | cache | Depth Pro | 10 | calibration | — | converged |
| `block3_metric3d_v2_s20` | cache | Metric3D v2 ViT-L | 20 | calibration | — | converged |
| `block3_unidepth_v2_s20` | cache | UniDepthV2 ViT-L | 20 | calibration | — | converged |
| `block3_depth_anything_v2_s20` | cache | Depth Anything V2 Large (inverse, scale only) | 20 | calibration | — | **alignment not converged (2-cycle)** |
| `block3_control_independent` | exact_independent | — | 20 | calibration | white 0.04445 (= Depth Pro s20 sigma_global) | converged |
| `block3_control_repeated` | exact_repeated | — | 20 | calibration | white 0.002, planted tau 2.0, rho 0.3 | **alignment not converged (2-cycle)** |

### Notes

- **Not converged.** Two runs cycle between two states in the label-free pair-graph rounds:
  - **exact_repeated:** one frame cycles, frame 56 (yaw 270, end of strip 26). It swings by about 0.20 in log scale each round.
  - **DAv2:** most frames cycle, by 0.2–0.3 in log scale. The scale-only inverse model probably does not fit DAv2's affine-invariant disparity.
- **A8.** The single-frame calibration fails on held-out land for every real model (s_z 1.2–1.8). It also fails for the repeated control. It passes for the independent control, whose fused held-out land s_z is 1.12 [1.05, 1.23] at b886adb (1.25 at def370d, before the record-weight fix).
- **Two folders sharing a name.** `synthetic_control_*` are the synthetic-block controls from the earlier commit. They are not block 3 results.
