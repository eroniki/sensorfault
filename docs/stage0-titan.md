# Stage 0 sizing on the TITAN RTX (2026-11-15)

Memory and step time of the Marigold-style SDXL LoRA step in `scripts/stage0_sizing.py`, measured with `scripts/stage0_grid.sh`. Raw JSON lines: `docs/stage0-titan.jsonl`; table made with `scripts/stage0_table.py`.

## Host and software

| Item | Value |
|---|---|
| GPU | NVIDIA TITAN RTX, 24576 MiB, Turing sm_75 (fp16 tensor cores, no bf16) |
| Driver / CUDA | 550.163.01 / 12.4 (toolkit 12.4.131) |
| Python / torch | 3.10.22 / 2.4.1+cu124 (cuDNN 9.1.0) |
| Harness packages | diffusers 0.30.3, peft 0.12.0, accelerate 0.34.2, transformers 4.44.2, numpy 1.26.4 |
| UNet config | `stabilityai/stable-diffusion-xl-base-1.0` `unet/config.json` (sha256 `30ebc707...2dbae6`), kept outside the repository; no weights downloaded |
| GPU state | idle before the grid (488 MiB used by other processes, 0% utilisation); nothing else allocated during the runs |

Environment details: `docs/env.md` (TITAN section). Tests: `PYTHONPATH=src:tests python -m pytest -q` gives 24 passed, 1 skipped (the ETH3D real-data test without `ETH3D_PIPES`), the same as on the 2070 host.

## What is measured

The harness builds SDXL's U-Net from its config with random weights (2.61B parameters), widens `conv_in` to 8 channels and trains it in full, adds LoRA on `to_q`, `to_k`, `to_v`, `to_out.0`, and runs 5 AdamW steps of a v-prediction loss with fp16 autocast and a grad scaler (or plain fp32 with `--fp32`). Columns:
- **peak alloc**: `torch.cuda.max_memory_allocated`, as printed by the harness;
- **peak device**: the largest `nvidia-smi` reading during the run (sampled every 200 ms) minus the idle reading before it. It includes the CUDA context and the allocator cache, so it is the number that has to fit;
- **ms/step**: median of the 5 steps;
- **fits 80%**: peak device at or below 80% of the card (0.8 x 24576 MiB = 20.6 GB).

## Results

| res | batch | rank | ckpt | precision | trainable params | peak alloc GB | peak device GB | ms/step | fits 80% (20.6 GB) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 1 | 16 | yes | fp16 autocast | 23.2 M | 11.12 | 12.00 | 873 | yes |
| 512 | 1 | 16 | no | fp16 autocast | 23.2 M | 17.37 | 17.77 | 470 | yes |
| 512 | 1 | 32 | yes | fp16 autocast | 46.5 M | 11.54 | 12.47 | 894 | yes |
| 512 | 1 | 32 | no | fp16 autocast | 46.5 M | 17.79 | 18.20 | 537 | yes |
| 512 | 1 | 64 | yes | fp16 autocast | 92.9 M | 12.37 | 13.48 | 859 | yes |
| 512 | 1 | 64 | no | fp16 autocast | 92.9 M | 18.64 | 19.13 | 477 | yes |
| 512 | 2 | 16 | yes | fp16 autocast | 23.2 M | 11.30 | 11.97 | 860 | yes |
| 512 | 2 | 16 | no | fp16 autocast | 23.2 M | 19.47 | 19.83 | 473 | yes |
| 512 | 2 | 32 | yes | fp16 autocast | 46.5 M | 11.71 | 12.45 | 871 | yes |
| 512 | 2 | 32 | no | fp16 autocast | 46.5 M | 19.90 | 20.26 | 477 | yes |
| 512 | 2 | 64 | yes | fp16 autocast | 92.9 M | 12.55 | 13.47 | 866 | yes |
| 512 | 2 | 64 | no | fp16 autocast | 92.9 M | 20.76 | 21.18 | 480 | no |
| 768 | 1 | 16 | yes | fp16 autocast | 23.2 M | 11.33 | 12.24 | 870 | yes |
| 768 | 1 | 16 | no | fp16 autocast | 23.2 M | 19.92 | 20.71 | 475 | no |
| 768 | 1 | 32 | yes | fp16 autocast | 46.5 M | 11.75 | 12.73 | 856 | yes |
| 768 | 1 | 32 | no | fp16 autocast | 46.5 M | 20.35 | 21.15 | 472 | no |
| 768 | 1 | 64 | yes | fp16 autocast | 92.9 M | 12.59 | 13.62 | 940 | yes |
| 768 | 1 | 64 | no | fp16 autocast | 92.9 M | 21.21 | 22.14 | 469 | no |
| 768 | 2 | 16 | yes | fp16 autocast | 23.2 M | 11.76 | 12.59 | 930 | yes |
| 768 | 2 | 16 | no | fp16 autocast | - | - | 24.8 | OutOfMemoryError | no |
| 768 | 2 | 32 | yes | fp16 autocast | 46.5 M | 12.14 | 13.12 | 931 | yes |
| 768 | 2 | 32 | no | fp16 autocast | - | - | 24.8 | OutOfMemoryError | no |
| 768 | 2 | 64 | yes | fp16 autocast | 92.9 M | 12.98 | 14.13 | 1001 | yes |
| 768 | 2 | 64 | no | fp16 autocast | - | - | 24.8 | OutOfMemoryError | no |
| 512 | 1 | 32 | yes | fp32 | 46.5 M | 11.56 | 12.41 | 686 | yes |

## Readings

- **With gradient checkpointing every setting fits**, at 12.0-14.1 GB on the device, at least 6.5 GB under the 80% line. The floor is the frozen base weights, which the harness stores in **fp32**: 2.567B x 4 bytes = 10.3 GB. On top of that, LoRA rank adds 0.4-0.8 GB per doubling, and going from 512 px batch 1 to 768 px batch 2 adds about 0.6 GB.
- **Without checkpointing** only 512 px fits (batch 1, and batch 2 up to rank 32). 768 px at batch 1 runs but exceeds the 80% line (20.7-22.1 GB), and 768 px at batch 2 runs out of memory.
- **Step time hardly depends on resolution or batch:** about 470 ms per step without checkpointing and 860-1000 ms with it, across 2.25x the pixels and 2x the batch. The fp32 run (686 ms) is faster than the same setting under fp16 autocast (894 ms). [Inference] At these sizes the step is limited by CPU-side and kernel-launch overhead, not by GPU arithmetic: one `nvidia-smi` sample taken mid-grid showed 27% utilisation. Per-sample cost therefore falls with batch size, e.g. 0.44 s per sample at 512 px batch 2 with checkpointing and 0.47 s at 768 px. Larger batches, CUDA graphs or `torch.compile` were not tried.
- **fp16 autocast saves no memory here** (11.54 GB vs 11.56 GB in fp32), because the frozen weights are fp32 in both runs.

## Where this differs from PLAN section 7

- PLAN assumes frozen **fp16** base weights (about 5 GB). The harness keeps them in fp32 (10.3 GB). Storing them in fp16 would lower every row by about 5.1 GB [EST, not measured] and might let 768 px run without checkpointing. That would need a harness change and a re-run; not done here.
- The step has **no VAE and no text encoders** (random latents, zero text embeddings). The fp32 VAE adds about 0.3 GB of weights plus encoder activations unless latents are pre-encoded.
- Random weights and 5 steps measure memory and time only. They say nothing about convergence or about fp16 overflow with real SDXL weights.

## Rough consequence for PLAN section 10 [EST]

From step time alone, a Marigold-scale stage 1 (18K iterations x batch 32 = 576K samples) at batch 2 with checkpointing would take about 70 h at 512 px and 75 h at 768 px on this card, before data loading and VAE encoding. It would be less if larger batches cut the per-sample time, as the overhead reading above suggests.
