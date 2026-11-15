"""Stage-0 sizing harness: memory and step time of a Marigold-style depth fine-tune (frozen backbone + LoRA + widened conv_in).

Measures, for a UNet2DConditionModel config, the trainable parameter count, peak CUDA memory and ms per optimiser step at a
given latent resolution, batch size, LoRA rank and precision. It does NOT download weights: random initialisation is enough for
memory and speed. Targets (author decision): SDXL base, LoRA, on the 24 GB TITAN RTX (Turing: fp16 autocast, no bf16).

  python scripts/stage0_sizing.py --tiny --res 256                               # logic check, any GPU (used in tests)
  python scripts/stage0_sizing.py --config path/to/sdxl_unet_config.json --res 512 --batch 1 --rank 32 --ckpt
(SDXL's unet/config.json is ungated; fetch just that file, e.g. from the stable-diffusion-xl-base-1.0 repo.)

Model of one step: latent of RGB (4 ch) concatenated with the noised depth latent (4 ch) -> 8-channel conv_in (weights duplicated
and halved, as Marigold), v-prediction loss on the depth latent, random timesteps. Text conditioning is zeros of the right shape.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from diffusers import DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

TINY = dict(sample_size=32, in_channels=4, out_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(32, 64), layers_per_block=1,
            cross_attention_dim=32, attention_head_dim=4, addition_embed_type="text_time", addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=8 * 6 + 32, transformer_layers_per_block=1, norm_num_groups=8)


def widen_conv_in(unet: UNet2DConditionModel) -> None:
    old = unet.conv_in
    new = torch.nn.Conv2d(old.in_channels * 2, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.copy_(torch.cat([old.weight, old.weight], 1) / 2)
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=old.in_channels * 2)


def build(cfg: dict, rank: int, ckpt: bool):
    unet = UNet2DConditionModel(**{k: v for k, v in cfg.items() if not k.startswith("_")})
    widen_conv_in(unet)
    for p in unet.parameters():
        p.requires_grad_(False)
    unet = get_peft_model(unet, LoraConfig(r=rank, lora_alpha=rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for p in unet.base_model.model.conv_in.parameters():      # the widened input conv is trained in full
        p.requires_grad_(True)
    if ckpt:
        unet.base_model.model.enable_gradient_checkpointing()
    return unet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--res", type=int, default=512, help="image resolution; latent = res / 8")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing")
    ap.add_argument("--fp32", action="store_true")
    a = ap.parse_args()
    cfg = TINY if a.tiny else json.load(open(a.config))
    dev = "cuda"
    unet = build(cfg, a.rank, a.ckpt).to(dev)
    sched = DDPMScheduler(num_train_timesteps=1000, prediction_type="v_prediction", beta_schedule="scaled_linear")
    params = [p for p in unet.parameters() if p.requires_grad]
    n_tr, n_all = sum(p.numel() for p in params), sum(p.numel() for p in unet.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=not a.fp32)
    h = a.res // 8
    cad = cfg["cross_attention_dim"]
    add_dim = cfg["projection_class_embeddings_input_dim"] - 6 * cfg["addition_time_embed_dim"]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(a.steps):
        rgb = torch.randn(a.batch, 4, h, h, device=dev)
        depth = torch.randn(a.batch, 4, h, h, device=dev)
        noise = torch.randn_like(depth)
        t = torch.randint(0, 1000, (a.batch,), device=dev)
        noisy = sched.add_noise(depth, noise, t)
        target = sched.get_velocity(depth, noise, t)
        cond = {"text_embeds": torch.zeros(a.batch, add_dim, device=dev), "time_ids": torch.zeros(a.batch, 6, device=dev)}
        enc = torch.zeros(a.batch, 77, cad, device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=not a.fp32):
            pred = unet(torch.cat([rgb, noisy], 1), t, encoder_hidden_states=enc, added_cond_kwargs=cond).sample
            loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        if not torch.isfinite(loss):
            raise SystemExit("non-finite loss")
    print(json.dumps({"params_total": n_all, "params_trainable": n_tr, "res": a.res, "batch": a.batch, "rank": a.rank,
                      "ckpt": a.ckpt, "fp16": not a.fp32, "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                      "ms_per_step_median": round(1000 * sorted(times)[len(times) // 2], 1), "loss_last": float(loss)}))


if __name__ == "__main__":
    main()
