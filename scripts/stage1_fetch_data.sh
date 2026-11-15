#!/usr/bin/env bash
# Fetch what stage 1 needs on a host that can reach the internet. Everything here is public; nothing needs a login.
#   scripts/stage1_fetch_data.sh sdxl DIR            SDXL base 1.0 diffusers files used for training (about 7.5 GB: fp16 U-Net, fp32 VAE, fp16 text encoders)
#   scripts/stage1_fetch_data.sh blendedmvs DIR      BlendedMVS low-res (113 scenes, about 27.5 GB zip; needs about 60 GB free while unzipping)
#   scripts/stage1_fetch_data.sh hypersim DIR [N]    first N training-split Hypersim scenes (default 12): per scene a 1-20 GB zip is downloaded, only the
#                                                    preview JPGs and depth_meters hdf5 files are kept, the zip is deleted
# Check free disk space first (df -h). Licences: BlendedMVS CC BY 4.0; Hypersim CC BY-SA 3.0 (Apple); SDXL CreativeML Open RAIL++-M
# (research use and fine-tuning allowed; do not distribute derived weights without the licence text and use restrictions).
set -euo pipefail
what=${1:?sdxl|blendedmvs|hypersim}
dir=${2:?target dir}
mkdir -p "$dir"
case "$what" in
  sdxl)
    python - "$dir" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", local_dir=sys.argv[1],
    allow_patterns=["model_index.json", "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors", "vae/config.json",
                    "vae/diffusion_pytorch_model.safetensors", "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
                    "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors", "tokenizer/*", "tokenizer_2/*", "scheduler/*"])
print("sdxl ready in", sys.argv[1])
PY
    ;;
  blendedmvs)
    cd "$dir"
    curl -L -C - -o BlendedMVS.zip https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0/BlendedMVS.zip
    unzip -q -n BlendedMVS.zip
    echo "unzipped; find the directory that holds <scene>/blended_images and point BLENDEDMVS_ROOT at it"
    ;;
  hypersim)
    n=${3:-12}
    cd "$dir"
    csv=metadata_images_split_scene_v1.csv
    [ -f $csv ] || curl -L -o $csv https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/$csv
    scenes=$(python - "$csv" "$n" <<'PY'
import csv, sys
seen = []
for r in csv.DictReader(open(sys.argv[1])):
    if r["split_partition_name"] == "train" and r["scene_name"] not in seen:
        seen.append(r["scene_name"])
print(" ".join(seen[: int(sys.argv[2])]))
PY
)
    for s in $scenes; do
      [ -d "$s/images" ] && { echo "have $s"; continue; }
      curl -L -C - -o "$s.zip" "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/$s.zip"
      unzip -q -o "$s.zip" "$s/images/scene_cam_*_final_preview/*" "$s/images/scene_cam_*_geometry_hdf5/frame.*.depth_meters.hdf5" || true
      rm -f "$s.zip"
      echo "kept $s: $(du -sh "$s" | cut -f1)"
    done
    echo "HYPERSIM_ROOT=$dir  HYPERSIM_SPLIT_CSV=$dir/$csv"
    ;;
  *) echo "unknown: $what" >&2; exit 2 ;;
esac
