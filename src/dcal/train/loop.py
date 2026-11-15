"""Stage-1 training loop: supervised fine-tune on affine-invariant depth with checkpoint/resume, logging and NaN guards."""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import MixedDepthDataset
from .model import build_models, load_trainable_state, make_schedulers, trainable_state


def gpu_used_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return round(float(out.stdout.strip().splitlines()[0]) / 1024, 2)
    except Exception:       # noqa: BLE001
        return None


def encode(vae, x, sf):
    """Latent mean scaled by the VAE scaling factor (deterministic encoding, as Marigold)."""
    with torch.no_grad():
        return vae.encode(x.to(next(vae.parameters()).dtype)).latent_dist.mean.float() * sf


def conditioning(cond, B, res, device):
    emb = cond["embeds"][None].expand(B, -1, -1).float()
    pooled = cond["pooled"][None].expand(B, -1).float()
    tid = torch.tensor([res, res, 0, 0, res, res], dtype=torch.float32, device=device)[None].expand(B, -1)
    return emb, {"text_embeds": pooled, "time_ids": tid}


def diffusion_loss(unet, vae, sched, cond, batch, cfg, device, amp, generator=None):
    sf = vae.config.scaling_factor
    rgb = batch["rgb"].to(device)
    depth3 = batch["depth"].to(device).repeat(1, 3, 1, 1)
    z_rgb = encode(vae, rgb, sf)
    z_d = encode(vae, depth3, sf)
    B = z_d.shape[0]
    noise = torch.randn(z_d.shape, device=device, generator=generator)
    t = torch.randint(0, sched.config.num_train_timesteps, (B,), device=device, generator=generator)
    noisy = sched.add_noise(z_d, noise, t)
    target = sched.get_velocity(z_d, noise, t) if sched.config.prediction_type == "v_prediction" else noise
    emb, added = conditioning(cond, B, rgb.shape[-1], device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        pred = unet(torch.cat([z_rgb, noisy], 1), t, encoder_hidden_states=emb, added_cond_kwargs=added).sample
    m = F.avg_pool2d(batch["valid"].to(device).float(), 8) >= 0.5            # latent-resolution validity mask
    m = m.float().expand_as(pred)
    return (((pred.float() - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)


@torch.no_grad()
def sample_depth(unet, vae, infer_sched, cond, rgb, device, amp, steps=10, seed=0):
    """Single-sample affine-invariant depth in [-1, 1] by DDIM (trailing spacing)."""
    sf = vae.config.scaling_factor
    z_rgb = encode(vae, rgb.to(device), sf)
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(z_rgb.shape, device=device, generator=g)
    infer_sched.set_timesteps(steps, device=device)
    emb, added = conditioning(cond, z.shape[0], rgb.shape[-1], device)
    for t in infer_sched.timesteps:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = unet(torch.cat([z_rgb, z], 1), t, encoder_hidden_states=emb, added_cond_kwargs=added).sample
        z = infer_sched.step(out.float(), t, z).prev_sample
    dec = vae.decode((z / sf).to(next(vae.parameters()).dtype)).sample.float()
    return dec.mean(1, keepdim=True).clamp(-1, 1)


def affine_absrel(pred, gt01, valid):
    """AbsRel after a least-squares scale/shift of the prediction to the ground truth (both in the same normalised units shifted to be positive)."""
    p, g = pred[valid].double(), gt01[valid].double()
    A = torch.stack([p, torch.ones_like(p)], 1)
    sol = torch.linalg.lstsq(A, g[:, None]).solution[:, 0]
    fit = A @ sol
    return float(((fit - g).abs() / g.clamp(min=1e-3)).mean())


def save_ckpt(path: Path, unet, opt, scaler, step, cfg, extra):
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "trainable": trainable_state(unet), "opt": opt.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                "cfg": cfg, "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                "extra": extra}, tmp)
    os.replace(tmp, path)


def latest_ckpt(workdir: Path):
    cks = sorted(workdir.glob("ckpt_*.pt"))
    return cks[-1] if cks else None


def prune(workdir: Path, keep_last=3, milestone_every=2000):
    cks = sorted(workdir.glob("ckpt_*.pt"))
    for c in cks[:-keep_last]:
        step = int(c.stem.split("_")[1])
        if milestone_every and step % milestone_every == 0:
            continue
        c.unlink()


class Stop(Exception):
    pass


def train(cfg: dict, workdir: str, resume: str = "auto", max_steps: int | None = None, device: str | None = None, stop_at: int | None = None):
    """max_steps shortens the whole run (including the LR schedule); stop_at only interrupts it after writing a checkpoint (used to test resume)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device == "cuda" and not cfg.get("fp32", False)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    res, mb, accum, total = cfg["res"], cfg["micro_batch"], cfg["accum"], cfg["total_steps"]
    if max_steps is not None:
        total = min(total, max_steps)
    unet, vae, cond = build_models(cfg, device)
    sched, infer_sched = make_schedulers(cfg)
    params_lora = [p for n, p in unet.named_parameters() if p.requires_grad and "conv_in" not in n]
    params_conv = [p for n, p in unet.named_parameters() if p.requires_grad and "conv_in" in n]
    opt = torch.optim.AdamW([{"params": params_lora, "lr": cfg["lr"]}, {"params": params_conv, "lr": cfg.get("lr_conv", cfg["lr"] / 3)}],
                            weight_decay=cfg.get("weight_decay", 0.01), betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=cfg.get("init_scale", 2.0 ** 12)) if amp else None
    step = 0
    rng_state = None
    extra = {"skipped": 0, "nonfinite_loss": 0, "wall_s": 0.0}
    ck = latest_ckpt(wd) if resume == "auto" else (Path(resume) if resume else None)
    if ck is not None and ck.exists():
        d = torch.load(ck, map_location="cpu", weights_only=False)
        load_trainable_state(unet, d["trainable"])
        opt.load_state_dict(d["opt"])
        if scaler and d["scaler"]:
            scaler.load_state_dict(d["scaler"])
        step, extra = d["step"], d["extra"]
        rng_state = d["rng"]["torch"]
        print(f"resumed from {ck} at step {step}", flush=True)
    samples_per_step = mb * accum
    ds = MixedDepthDataset(cfg["datasets"], "train", res, seed, total * samples_per_step)
    val = MixedDepthDataset(cfg["datasets"], "val", res, seed + 1, cfg.get("val_samples", 16)) if cfg.get("val_samples", 16) else None
    start_k = step * samples_per_step
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, range(start_k, len(ds))), batch_size=mb, shuffle=False,
                                         num_workers=cfg.get("workers", 4), drop_last=True, persistent_workers=cfg.get("workers", 4) > 0)
    it = iter(loader)
    if rng_state is not None:          # after the iterator exists: creating it draws from the global RNG, exactly as in an uninterrupted run
        torch.set_rng_state(rng_state)
    log_path = wd / "log.jsonl"
    logf = open(log_path, "a")
    stop = {"flag": False}

    def handler(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    last_ckpt_t = time.time()
    window, t_win = [], time.time()
    consecutive_bad = 0
    unet.train()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    while step < total:
        warm = min(1.0, (step + 1) / max(cfg.get("warmup", 100), 1))
        decay = 0.5 * (1 + math.cos(math.pi * step / max(total, 1))) if cfg.get("cosine", True) else 1.0
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * warm * (cfg.get("min_lr_frac", 0.05) + (1 - cfg.get("min_lr_frac", 0.05)) * decay)
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        bad = False
        for _ in range(accum):
            batch = next(it)
            loss = diffusion_loss(unet, vae, sched, cond, batch, cfg, device, amp)
            if not torch.isfinite(loss):
                bad = True
                extra["nonfinite_loss"] += 1
                continue
            (scaler.scale(loss / accum) if scaler else loss / accum).backward()
            tot_loss += float(loss) / accum
        gnorm = float("nan")
        skipped = False
        if bad:
            consecutive_bad += 1
            skipped = True
            opt.zero_grad(set_to_none=True)
            if consecutive_bad >= cfg.get("max_consecutive_bad", 10):
                raise RuntimeError(f"{consecutive_bad} consecutive non-finite losses at step {step}: stopping (fp16 overflow or divergence)")
        else:
            consecutive_bad = 0
            if scaler:
                scaler.unscale_(opt)
            gnorm = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], cfg.get("max_grad_norm", 1.0)))
            if scaler:
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                skipped = scaler.get_scale() < before
            else:
                if math.isfinite(gnorm):
                    opt.step()
                else:
                    skipped = True
            if skipped:
                extra["skipped"] += 1
        step += 1
        window.append(tot_loss)
        if step % cfg.get("log_every", 10) == 0 or step == total:
            dt = time.time() - t_win
            rec = {"step": step, "loss": float(np.mean(window)), "lr": opt.param_groups[0]["lr"], "grad_norm": gnorm,
                   "scale": scaler.get_scale() if scaler else None, "skipped_total": extra["skipped"], "nonfinite_loss_total": extra["nonfinite_loss"],
                   "s_per_step": dt / len(window), "samples_per_s": len(window) * samples_per_step / dt,
                   "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None, "gpu_used_gb": gpu_used_gb() if device == "cuda" else None}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            window, t_win = [], time.time()
        if val is not None and cfg.get("eval_every") and (step % cfg["eval_every"] == 0 or step == total):
            rec = evaluate(unet, vae, sched, infer_sched, cond, val, cfg, device, amp)
            rec["step"] = step
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            print(json.dumps(rec), flush=True)
            unet.train()
        due = (stop_at is not None and step == stop_at) or (time.time() - last_ckpt_t) / 60 >= cfg.get("ckpt_every_min", 60) or (cfg.get("ckpt_every_steps") and step % cfg["ckpt_every_steps"] == 0)
        if due or stop["flag"] or step == total:
            save_ckpt(wd / f"ckpt_{step:07d}.pt", unet, opt, scaler, step, cfg, extra)
            prune(wd, cfg.get("keep_last", 3), cfg.get("milestone_every", 2000))
            last_ckpt_t = time.time()
            print(f"checkpoint at step {step}", flush=True)
        if stop["flag"]:
            print("stop signal received: checkpoint written, exiting", flush=True)
            break
        if stop_at is not None and step == stop_at:
            break
    logf.close()
    return {"step": step, "workdir": str(wd), "extra": extra}


@torch.no_grad()
def evaluate(unet, vae, sched, infer_sched, cond, val, cfg, device, amp):
    unet.eval()
    g = torch.Generator(device=device).manual_seed(1234)
    losses = []
    n = len(val)
    for i in range(0, n, cfg["micro_batch"]):
        b = default_collate([val[k] for k in range(i, min(i + cfg["micro_batch"], n))])
        losses.append(float(diffusion_loss(unet, vae, sched, cond, b, cfg, device, amp, generator=g)))
    rec = {"val_loss": float(np.mean(losses))}
    k = cfg.get("val_sample_images", 4)
    if k:
        b = default_collate([val[j] for j in range(min(k, n))])
        pred = sample_depth(unet, vae, infer_sched, cond, b["rgb"], device, amp, steps=cfg.get("val_ddim_steps", 10))
        gt = (b["depth"].to(device) + 1) / 2 + 0.05             # positive "depth-like" values in [0.05, 1.05]
        pr = (pred + 1) / 2 + 0.05
        valid = b["valid"].to(device)[:, 0:1]
        rec["val_affine_absrel"] = float(np.mean([affine_absrel(pr[i], gt[i], valid[i]) for i in range(pr.shape[0])]))
        rec["val_finite_samples"] = bool(torch.isfinite(pred).all())
    return rec


def default_collate(items):
    return {"rgb": torch.stack([x["rgb"] for x in items]), "depth": torch.stack([x["depth"] for x in items]),
            "valid": torch.stack([x["valid"] for x in items])}
