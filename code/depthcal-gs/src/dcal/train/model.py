"""Models for stage 1: SDXL U-Net (frozen, fp16) + LoRA + widened 8-channel conv_in, a frozen VAE, and empty-prompt conditioning.

Real weights: diffusers-format SDXL base (a local directory or the Hugging Face id stabilityai/stable-diffusion-xl-base-1.0; ungated).
`tiny=True` builds small random models of the same structure so the whole loop can be tested without weights.
"""
from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

TINY_UNET = dict(sample_size=16, in_channels=4, out_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                 up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(32, 64), layers_per_block=1,
                 cross_attention_dim=32, attention_head_dim=4, addition_embed_type="text_time", addition_time_embed_dim=8,
                 projection_class_embeddings_input_dim=8 * 6 + 24, transformer_layers_per_block=1, norm_num_groups=8)
TINY_VAE = dict(in_channels=3, out_channels=3, latent_channels=4, down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 16, 16), layers_per_block=1, norm_num_groups=4,
                scaling_factor=0.13025)
TINY_POOLED = 24


def make_schedulers(cfg: dict):
    kw = dict(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
              prediction_type=cfg.get("prediction_type", "v_prediction"), rescale_betas_zero_snr=cfg.get("zero_snr", True))
    train = DDPMScheduler(**kw)
    infer = DDIMScheduler(**kw, timestep_spacing="trailing", clip_sample=False, set_alpha_to_one=True)
    return train, infer


def widen_conv_in(unet: UNet2DConditionModel) -> None:
    """Duplicate the input conv over the RGB-latent channels and halve it (Marigold): the model starts as the pretrained one."""
    old = unet.conv_in.to(torch.float32)
    new = torch.nn.Conv2d(old.in_channels * 2, old.out_channels, old.kernel_size, old.stride, old.padding)
    with torch.no_grad():
        new.weight.copy_(torch.cat([old.weight, old.weight], 1) / 2)
        new.bias.copy_(old.bias)
    unet.conv_in = new
    unet.register_to_config(in_channels=old.in_channels * 2)


def empty_prompt_embeds(path: str, device, dtype=torch.float16):
    """SDXL conditioning for the empty prompt: (77, 2048) hidden states and (1280,) pooled embedding, from both CLIP encoders."""
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    tok1 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(path, subfolder="text_encoder", torch_dtype=dtype, variant="fp16").to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(path, subfolder="text_encoder_2", torch_dtype=dtype, variant="fp16").to(device).eval()
    with torch.no_grad():
        ids1 = tok1("", padding="max_length", max_length=tok1.model_max_length, return_tensors="pt").input_ids.to(device)
        ids2 = tok2("", padding="max_length", max_length=tok2.model_max_length, return_tensors="pt").input_ids.to(device)
        h1 = te1(ids1, output_hidden_states=True).hidden_states[-2]
        o2 = te2(ids2, output_hidden_states=True)
        h2 = o2.hidden_states[-2]
        emb = torch.cat([h1, h2], -1)[0].float().cpu()
        pooled = o2.text_embeds[0].float().cpu()
    del te1, te2
    return emb, pooled


def build_models(cfg: dict, device: str):
    """Returns (unet_peft, vae, cond) with cond = dict(embeds (77,D), pooled (P,), res)."""
    rank = cfg.get("rank", 32)
    if cfg.get("tiny"):
        unet = UNet2DConditionModel(**TINY_UNET)
        vae = AutoencoderKL(**TINY_VAE)
        embeds = torch.zeros(77, TINY_UNET["cross_attention_dim"])
        pooled = torch.zeros(TINY_POOLED)
    else:
        path = cfg["pretrained"]
        unet = UNet2DConditionModel.from_pretrained(path, subfolder="unet", torch_dtype=torch.float16, variant=cfg.get("unet_variant", "fp16"))
        vp = cfg.get("vae", "fp32")
        if vp == "fp32":
            vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
        else:                           # an fp16-safe VAE (for example madebyollin/sdxl-vae-fp16-fix), run in fp16
            vae = AutoencoderKL.from_pretrained(vp, torch_dtype=torch.float16)
        cache = Path(cfg.get("embeds_cache", "empty_prompt_embeds.pt"))
        if cache.exists():
            d = torch.load(cache, map_location="cpu")
            embeds, pooled = d["embeds"], d["pooled"]
        else:
            embeds, pooled = empty_prompt_embeds(path, device)
            torch.save({"embeds": embeds, "pooled": pooled}, cache)
    widen_conv_in(unet)
    for p in unet.parameters():
        p.requires_grad_(False)
    if cfg.get("tiny"):
        unet.float()
    else:
        for n, p in unet.named_parameters():
            if not n.startswith("conv_in"):
                p.data = p.data.half()
    unet = get_peft_model(unet, LoraConfig(r=rank, lora_alpha=cfg.get("lora_alpha", rank), lora_dropout=0.0,
                                           target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for p in unet.base_model.model.conv_in.parameters():
        p.requires_grad_(True)
    if cfg.get("grad_ckpt", True):
        unet.base_model.model.enable_gradient_checkpointing()
    vae.requires_grad_(False)
    vae.eval()
    unet.to(device)
    vae.to(device)
    cond = {"embeds": embeds.to(device), "pooled": pooled.to(device)}
    return unet, vae, cond


def trainable_state(unet) -> dict:
    return {n: p.detach().cpu() for n, p in unet.named_parameters() if p.requires_grad}


def load_trainable_state(unet, state: dict) -> None:
    params = dict(unet.named_parameters())
    missing = [k for k in state if k not in params]
    if missing:
        raise KeyError(f"checkpoint has parameters the model lacks: {missing[:3]}")
    with torch.no_grad():
        for k, v in state.items():
            params[k].copy_(v.to(params[k].dtype))
