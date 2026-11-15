# Stage 1 on the TITAN RTX: supervised fine-tune of SDXL + LoRA for affine-invariant depth

Status: **code and config written and tested locally; smoke run and full run not started yet** (the TITAN session fills the run record at the bottom).

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
- **Length:** `total_steps` = 10,000 (320K samples; Marigold used 576K) in `configs/stage1_sdxl_lora.json`. Expected wall time [EST, before the smoke run measures it]: about 40-50 h on the TITAN RTX (stage 0: roughly 0.85 s per micro-batch of 2 at 512 px with checkpointing, plus the fp32 VAE encodes). If the smoke run says more than about 72 h, reduce `total_steps` and write the number here.

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

## Run record (to be filled by the TITAN session)

| Item | Value |
|---|---|
| Smoke run date, verdict, s/step, peak memory | |
| BlendedMVS depth convention verdict | |
| Datasets actually used (scenes/images) | |
| Full run start (date and time, host) | |
| Config (commit, overrides) | |
| total_steps, s/step measured, **expected length** | |
| Checkpoint directory and newest checkpoint | |
| tmux session / pid file | |
| Notes (disk, throughput, incidents) | |
