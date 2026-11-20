# Shadow check results

`experiments/shadow_check.py` output, one folder per model: `summary.md`, `summary.json`, `part2_entities.json`,
`part2_costs.npz` (cost curves), and the panels `panels_part1_steps.png` and `panels_part2.png`.

## depth_pro (step 4, 2026-11-19, host 3101, CPU only, 15 min)

Command: `PYTHONPATH=src python experiments/shadow_check.py --model depth_pro --step 4 --out results/shadow_check/depth_pro`.
The branch is based on `stage-a/pipeline`, which holds the current Stage A code (records weights, README); main does not have it yet.

### Headline (primary mask: 31 px window, threshold 0.6; label-free alignment)

- **Shadows are rare.** The median shadow share of flat ground per frame is 1.0%. It is 4.2% with a 61 px window and 0.24% with threshold 0.5.
- **(a) Step across the edge, mean |dark side minus bright side| within 5 px.** Shadow boundaries: 0.40 m [0.38, 0.42]. Non-shadow luminance edges of matched contrast: 0.32 m [0.30, 0.33]. No-edge ground: 0.26 m [0.24, 0.28]. The signed shadow step is -0.02 m [-0.07, 0.02], so there is no consistent sign. More than 10 px from real height steps, the shadow step is 0.36 m against 0.30 m for edges. The same ordering holds for both alternative masks.
- **(b) Shadow minus lit on the same 8 m entity.** +0.025 m [-0.04, 0.09] over 693 entities, and 47% of entities are lowered. Shadowed ground is neither raised nor lowered.
- **(c) Spearman correlation of entity |dh| with boundary-band share.** 0.08 [-0.10, 0.22]. It is 0.01 for dh_loc.
- **Repetition on 1 m cells, dh_loc.** dh_loc is the frame's 64 m tile median removed, which takes out the per-frame scale error.

  | cells | rho, all pairs | null | rho, same heading | null |
  |---|---|---|---|---|
  | band | 0.33 [0.28, 0.40] | 0.02 | 0.51 [0.44, 0.59] | 0.02 |
  | other | 0.36 [0.30, 0.42] | 0.01 | 0.63 [0.55, 0.71] | 0.01 |

  Errors at shadow boundaries are strongly repeated, but no more than other flat ground.
- **Part 2.** Only 19 entities pass the faulty-shadow selection, against a target of 200. Their Depth Pro plane offsets are -42 to +19 m, all outside the +-6 m sweep. The minimum of the photometric cost lies at a median |nu_photo| of 0.3 m, and at 0.1 m for 19 texture-matched controls. Using the local offset nu_depth_loc (median 1.2 m), the pixels are closer to the truth in 16/19 faulty entities and 19/19 controls. The cost gradient at nu_depth_loc points toward the truth in 89% of faulty entities and 100% of controls. Median stiffness is 0.57 for faulty entities and 1.1 for controls. Its Spearman correlation with texture is 0.31 for faulty entities and 0.43 for controls.

### Caveats

- MatrixCity lighting is static and baked in, so a shadow sits at the same place in every frame and photo-consistency holds inside it.
- The faulty entities lie in building shadows next to tall facades (see the panels). Their large errors are confounded with proximity to buildings, and shadow cannot be separated from that here. The "more than 10 px from real steps" column is the only control for it.
- Depth Pro's flat-ground height error is large: rms 14.6 m with label-free alignment and 3.6 m after the tile median is removed. This dwarfs the 0.1 m effect at shadow edges.
- nu_photo has a small positive bias of +0.1 to +0.3 m, seen in the controls too. It is not investigated. Candidates are the half-pixel principal-point convention or the plane fit.
- The homography assumes a planar entity. Flat ground is restricted to normals within 10 deg of vertical and less than 0.5 m height range in 7 x 7.
- ZNCC is weak on low texture. The stiffness of the cost at its minimum tracks texture.
