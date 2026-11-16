# Stage 1 on the TITAN RTX: supervised fine-tune of SDXL + LoRA for affine-invariant depth

Status: **smoke run passed on the TITAN (2026-11-15); the first full-run segment ran to step 2,000 and stopped there as planned (2026-11-16 04:07 +03)** (`--stop-at 2000`, the author's decision after the smoke run). Continuing past 2,000 needs the author's go-ahead. Run record at the bottom.

## What is trained

- **Model:** SDXL base 1.0 U-Net, frozen, weights stored in fp16; LoRA (rank 32, alpha 32) on `to_q/to_k/to_v/to_out.0`; the input conv widened to 8 channels (RGB latent + noisy depth latent, weights duplicated and halved as in Marigold) and trained in full. Empty-prompt conditioning from both CLIP encoders, computed once and cached. VAE frozen, run in **fp32** (the plan's "VAE in FP32" option: SDXL's VAE can overflow in fp16; an fp16-safe VAE can be set with `"vae": "<id>"` if throughput needs it).
- **Target:** affine-invariant depth: per image (after cropping), depth normalised with its 2nd/98th percentiles to [-1, 1] and clipped (Marigold Eq. 3); replicated to 3 channels and encoded by the VAE. No metric scale, no camera embedding.
- **Objective:** v-prediction with zero-terminal-SNR betas (`prediction_type`, `zero_snr` in the config), uniform timesteps, MSE on the depth latent, **masked** at latent resolution by where the ground truth is valid (an 8x8 block counts if at least half its pixels are valid). Invalid pixels are filled with the nearest valid target value before encoding so that holes do not become edges.
- **Optimiser:** AdamW, LoRA lr 1e-4, conv_in lr 3e-5, 500 warm-up steps then cosine to 5% of the peak, weight decay 0.01, grad clip 1.0, fp16 autocast with a dynamic grad scaler (initial 4096), gradient checkpointing.
- **Batch:** micro-batch 4 x 8 accumulation = 32 images per optimiser step at **512 x 512** (random square crop of 60-100% of the area, horizontal flip, resize; bilinear for RGB, nearest for depth).
- **Data (ETH3D is never read; it stays held out):**
  - **BlendedMVS** (113 scenes, blended images + rendered `.pfm` depth), weight 0.6; the last 6 scenes (sorted by id) are held out for validation. The BlendedMVS+/++ volumes (the other BlendedMVG scenes) are not included in the first run.
  - **Hypersim** subset (first N training-split scenes from the official split file; preview JPG + `depth_meters` converted from ray distance to planar depth with the nominal focal length 886.81 px, per-scene tilt-shift ignored [assumption]), weight 0.4; its validation split is used for validation.
  - TartanAir: reader exists (`TartanAirDataset`, untested on real data) but it is **not** in the first run.
- **Length:** `total_steps` = 10,000 (320K samples; Marigold used 576K) in `configs/stage1_sdxl_lora.json`. Expected wall time [EST, before the smoke run measures it]: about 40-50 h on the TITAN RTX (stage 0: roughly 0.85 s per micro-batch of 2 at 512 px with checkpointing, plus the fp32 VAE encodes). If the smoke run says more than about 72 h, reduce `total_steps` and write the number here. **Measured:** 13.1 s per optimiser step at batch 32 (about 36-37 h for 10,000 steps). **First segment: stop at step 2,000** with a checkpoint (`TRAIN_ARGS="--stop-at 2000"`), keeping `total_steps` = 10,000 so that the cosine schedule is unchanged and a go-ahead resumes the same run. At step 2,000 the learning rate is still near its peak (about 9.5e-5), so validation there is a trend signal, not a finished model.

## Files

`configs/stage1_sdxl_lora.json` (full), `configs/stage1_smoke.json` (200 steps of batch 16), `src/dcal/train/{data,model,loop}.py`, `scripts/stage1_train.py`, `scripts/stage1_smoke.sh`, `scripts/stage1_smoke_check.py`, `scripts/stage1_run.sh`, `scripts/stage1_fetch_data.sh`, `scripts/check_blendedmvs_depth.py`, `tests/test_stage1.py`.

## Tested here (RTX 2070, 8 GB; not the TITAN)

- `tests/test_stage1.py` (CPU, tiny random model, synthetic data): data readers, affine-invariant target, loss masking, **resume reproduces an uninterrupted run to 1e-6**, NaN guard, loss falls on a learnable synthetic task, the z-depth/range check.
- Real SDXL weights, 36 steps at 256 px, batch 1 x 2, synthetic data: loads and trains, losses finite, grad scale stable at 4096, no skipped steps, validation sampling finite, 6.9 GB peak allocation, checkpoint 559 MB (LoRA + conv_in + Adam state), **resume from step 30 works**. This says nothing about depth quality: the loss on 36 noisy steps is flat.

## Procedure (TITAN session)

1. Fetch the bundle, check it out, run `PYTHONPATH=src:tests python -m pytest -q`. Install `h5py` (and `scipy`, `pillow`) if missing.
2. Data and weights: `scripts/stage1_fetch_data.sh sdxl|blendedmvs|hypersim DIR [N]` (all public, no login). Check free disk first; BlendedMVS is a split zip (15 parts of 1.95 GB plus a 282 MB final segment, 29.5 GB; the script downloads, joins and unzips them) and needs about 90 GB while joining and unzipping; Hypersim downloads one 1-20 GB zip per scene and keeps only the preview JPGs and depth hdf5. If space or bandwidth is short, start with BlendedMVS plus fewer Hypersim scenes and say so.
3. `python scripts/check_blendedmvs_depth.py <BlendedMVS>/<scene>`: reports whether the rendered depth is z-depth or ray distance. **Report the verdict before the full run.** (If it is ray distance, the BlendedMVS loader needs a conversion; the smoke run is unaffected.)
4. Smoke: `STAGE1_WORK=runs/smoke scripts/stage1_smoke.sh` with `SDXL_DIR`, `BLENDEDMVS_ROOT`, `HYPERSIM_ROOT`, `HYPERSIM_SPLIT_CSV` set. Passes if `scripts/stage1_smoke_check.py` prints `SMOKE PASSED`: loss falls, no non-finite losses, at most 10% skipped steps, grad scale not collapsed, finite validation, two checkpoints, a resume with step continuity. Note s/step and peak memory.
5. If it passes: launch `STAGE1_WORK=runs/stage1 scripts/stage1_run.sh` (tmux or nohup; resumes from the newest checkpoint; `kill -TERM $(cat runs/stage1/pid)` checkpoints and exits). One heavy job at a time: the script refuses with less than 16 GB free on the GPU (another process or another run); do not set `FORCE=1` without asking. Checkpoints: every 500 steps or 90 minutes, newest 3 kept plus every 2000 steps; metrics in `runs/stage1/log.jsonl` (loss, lr, grad norm, scale, skipped steps, s/step, peak allocation, device memory); validation every 500 steps.
6. Fill the run record below, commit on a branch, and return a bundle over Magic Wormhole. Rent nothing.

## Known risks

- Switching an epsilon-prediction SDXL to v-prediction with zero-terminal-SNR may need more steps than Marigold needed for the v-model SD 2; the loss falling in the smoke run does not show that the sampler works. `val_affine_absrel` (4-8 validation images, 10 DDIM steps) is the first usable signal.
- LoRA may fit less well than the full fine-tune Marigold used.
- BlendedMVS images are blended composites (not photographs) and its depth convention is unverified (step 3); Hypersim uses assumed nominal intrinsics; Hypersim's CC BY-SA terms matter only if weights or derived data are ever released.
- The validation set is small (32 samples).

## Run record (TITAN session)

| Item | Value |
|---|---|
| Smoke run date, verdict, s/step, peak memory | 2026-11-15, **SMOKE PASSED** (all 9 checks). 7.28-7.45 s/step at batch 16 (4 x 4, 512 px), 2.15-2.20 samples/s, 1510 s wall. Peak 8.94 GB allocated / 9.68 GB on the device (11.34 / 12.58 GB in the window that includes validation sampling). Loss 0.857 (step 45) to 0.472 (step 200); grad norm 0.19-0.46; grad scale 4096 throughout; 0 skipped, 0 non-finite. Validation: step 100 val_loss 0.529, val_affine_absrel 0.872; step 200 val_loss 0.528, val_affine_absrel 0.838. **Smoke-only overrides:** an 8-scene BlendedMVS subset with `val_scenes` 2 (558 train / 253 val pairs) and Hypersim ai_001_001-005 + val ai_005_005, ai_006_007; the full run uses the config default `val_scenes` 6 on the full set. Resume test: at fe4dc37 the CLI did not pass `--stop-at` (fixed in d0e8b01), so leg 1 was stopped with SIGTERM at step 109 and leg 2 resumed from `ckpt_0000109.pt` to 200 (the stop path the full run uses) |
| BlendedMVS depth convention verdict | **z-depth** (`check_blendedmvs_depth.py`, 12 pairs per scene): 5a3ca9cb270f0e3f14d0eddb z 0.00115 vs range 0.0707; 5bf03590d4392319481971dc z 0.139 vs range 0.395 (higher errors on that scene, z still about 3x better). No conversion needed |
| Datasets actually used (scenes/images) | BlendedMVS low-res v1.0.0 (16 split parts size-checked against the release, joined, unzip CRC OK): 113 scenes, 17,819 pairs; train 107 scenes / 17,116 pairs, validation the last 6 scenes / 703 pairs. Hypersim: 12 training-split scenes ai_001_001-ai_001_009 and ai_002_001-003 (1,991 pairs); validation-split ai_005_005, ai_006_007, ai_030_008 (257 pairs). ETH3D not read |
| Full run start (date and time, host) | 2026-11-15 20:02:36 +03, TITAN RTX host. A launch at 20:02:09 exited at config load (the launch copy's comment contained a literal `${...}`, which the config expander rejects); nothing was trained |
| Config (commit, overrides) | Branch `titan/stage1c` at 00dd71a (main d0e8b01 + a `TRAIN_ARGS` passthrough in `stage1_run.sh`). `configs/stage1_sdxl_lora.json` unchanged except that its environment-variable paths are resolved in a launch copy kept outside the repo: a detached tmux session takes the tmux server's environment, not the caller's. Interpreter `python -s` (user site off). `TRAIN_ARGS="--stop-at 2000"` |
| total_steps, s/step measured, **expected length** | `total_steps` 10,000 (LR schedule over 10k unchanged); **stops at step 2,000** (the author's decision). 13.1 s/step at step 10 (batch 32, 2.44 samples/s), then 14.2-14.6 s/step from step 500 on. **Stopped cleanly at step 2,000 on 2026-11-16 04:07:13 +03**; training wall time 29,055 s (8.07 h, validation included). At 14.4 s/step the full 10k would be about 40 h |
| Validation at 500 / 1,000 / 1,500 / 2,000 | val_loss 0.4385 / 0.3927 / 0.3510 / **0.3372**; val_affine_absrel 0.7193 / 0.6482 / 0.6409 / **0.6175**; samples finite at every check. Train loss (10-step window) 0.427 / 0.337 / 0.330 / 0.295; grad scale 4096 throughout, 8192 at step 2,000; 0 skipped, 0 non-finite steps. The learning rate at 2,000 is 9.1e-5 (cosine over 10k), so these are a trend, not a finished model |
| Checkpoint directory and newest checkpoint | outside the repo: `~/workspace/depthcal-runs/stage1/` (checkpoint every 500 steps or 90 min, newest 3 kept plus every 2,000 steps). **Final: `ckpt_0002000.pt`** (559 MB, written 04:06:57); also kept: `ckpt_0001500.pt`, `ckpt_0001865.pt` |
| tmux session / pid file | tmux session `depthcal-stage1`. **The `pid` file holds the bash subshell (2244456), not Python (2244457)**, so `kill -TERM $(cat pid)` would not stop training cleanly; stop this run early with `kill -TERM 2244457` (checkpoint, then exit) |
| Resume after a go-ahead | the same environment (`SDXL_DIR`, `STAGE1_WORK` = the workdir, `CONFIG` = the launch copy, `PY="python -s"`) and **no `TRAIN_ARGS`**: `scripts/stage1_run.sh` resumes from `ckpt_0002000.pt` to step 10,000 |
| Notes (disk, throughput, incidents) | SDXL from Hugging Face, anonymous (a stale stored token had caused a 401). BlendedMVS from the GitHub release at about 50-100 kB/s per connection, so 16 parallel connections; the fetch script had fetched only the last split part (fixed in aa8acb3). Hypersim from Apple's CDN; validation scenes had been missing (fixed in fe4dc37). Training allocates 8.9 GB (9.7 GB on the device); after the first validation the process kept 18.2 GB on the device (allocator cache from validation sampling, peak allocation 16.4 GB), so `torch.cuda.empty_cache()` after `evaluate()` would free about 9 GB for other GPU users. One heavy job at a time on the shared GPU. A CPU-only render of the denoising sequence from `ckpt_0001500.pt` (16 threads, about 20 min) slowed training by about 6% while it ran |

## First look at the checkpoints on held-out ETH3D `pipes` (2026-11-16; three views of one scene, a sanity check and not a result)

`scripts/stage1_sample.py` (LoRA + conv_in from the extracted trainable weights, 5 DDIM samples per image, scale/shift fitted to the sparse laser depth, AbsRel in real depth units; the untrained base gives AbsRel 0.38 / delta1 0.35 on view 6). Figures: `figures/peek/` (step 1,000) and `figures/peek_step2000/`.

| checkpoint, sampler | view 2 AbsRel | view 6 | view 10 | mean |
|---|---|---|---|---|
| step 1,000, 10 DDIM steps | 0.065 | 0.108 | 0.133 | 0.102 |
| step 2,000, 1 step | 0.052 | 0.119 | 0.144 | 0.105 |
| step 2,000, 4 steps | 0.049 | 0.125 | 0.147 | 0.107 |
| step 2,000, 10 steps | 0.049 | 0.132 | 0.148 | 0.110 |
| step 2,000, 30 steps | 0.050 | 0.136 | 0.148 | 0.111 |

- Step 2,000 is **not clearly better than step 1,000** on this scene (view 2 improves, views 6 and 10 get slightly worse); with three views the differences are within what I would call noise. Validation loss kept falling (0.3927 -> 0.3372).
- **Fewer sampler steps score better**: the 1-step estimate (the posterior-mean-like prediction from noise) beats 10 and 30 steps on views 6 and 10, and the median of 5 samples beats a single sample (single-sample AbsRel 0.055-0.158 against 0.052-0.148 for the median). The sample spread grows with the number of steps. This matches the TITAN session's observation on two validation images (1-step x0 better than a 30-step sample). For the calibration goal this is a real tension: a calibrated covariance needs genuine diverse samples, while the mean-like prediction is the more accurate one; the sampler (zero-terminal-SNR v-prediction with trailing DDIM) and more training are the first things to examine.
- Sample spread is highest at edges, the ceiling and the metal cabinet, where the error is also highest (visual agreement only; the correlation has not been measured).
