#!/usr/bin/env bash
# Fetch what stage 1 needs on a host that can reach the internet. Everything here is public; nothing needs a login.
#   scripts/stage1_fetch_data.sh sdxl DIR            SDXL base 1.0 diffusers files used for training (about 7.5 GB: fp16 U-Net, fp32 VAE, fp16 text encoders)
#   scripts/stage1_fetch_data.sh blendedmvs DIR      BlendedMVS low-res (113 scenes): the v1.0.0 release is a split zip, BlendedMVS.z01-z15 (1.95 GB each) +
#                                                    BlendedMVS.zip (282 MB) = 29.5 GB; the parts are downloaded (resumable, 3 at a time), joined, unzipped;
#                                                    needs about 90 GB free while joining and unzipping (parts, joined zip, scenes)
#   scripts/stage1_fetch_data.sh hypersim DIR [N] [V]  first N training-split Hypersim scenes (default 12) plus V validation-split scenes (default 2, the
#                                                    smallest by the number of frames: Hypersim splits by scene, so train scenes have no val frames): per scene a
#                                                    1-20 GB zip is downloaded, only the preview JPGs and depth_meters hdf5 files are kept, the zip is deleted
# Check free disk space first (df -h). Licences: BlendedMVS CC BY 4.0; Hypersim CC BY-SA 3.0 (Apple); SDXL CreativeML Open RAIL++-M
# (research use and fine-tuning allowed; do not distribute derived weights without the licence text and use restrictions).
set -euo pipefail
what=${1:?sdxl|blendedmvs|hypersim}
dir=${2:?target dir}
mkdir -p "$dir"
case "$what" in
  sdxl)
    # a stale Hugging Face token on the host makes the Hub reject even public downloads: ignore stored tokens (the repo is public)
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1 env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN python - "$dir" <<'PY'
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
    base=https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.0
    parts=$(printf 'BlendedMVS.z%02d ' $(seq 1 15))
    fetch() { curl -L -C - --retry 20 --retry-delay 10 --retry-all-errors -sS -o "$1" "$base/$1"; }
    export -f fetch; export base
    # three at a time; each part is resumable, so rerunning the script continues where it stopped
    printf '%s\n' $parts BlendedMVS.zip | xargs -P 3 -I{} bash -c 'fetch {}'
    sizes_ok=1
    for f in $parts; do [ "$(stat -c %s "$f")" = 1950000000 ] || { echo "incomplete: $f"; sizes_ok=0; }; done
    [ "$(stat -c %s BlendedMVS.zip)" = 281708973 ] || { echo "incomplete: BlendedMVS.zip"; sizes_ok=0; }
    [ $sizes_ok = 1 ] || { echo "re-run the script to resume the incomplete parts"; exit 4; }
    [ -f BlendedMVS_joined.zip ] || zip -s 0 BlendedMVS.zip --out BlendedMVS_joined.zip
    unzip -q -n BlendedMVS_joined.zip
    echo "unzipped; find the directory that holds <scene>/blended_images and point BLENDEDMVS_ROOT at it; the parts and the joined zip can be deleted"
    ;;
  hypersim)
    n=${3:-12}
    cd "$dir"
    csv=metadata_images_split_scene_v1.csv
    [ -f $csv ] || curl -L -o $csv https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/$csv
    v=${4:-2}
    scenes=$(python - "$csv" "$n" "$v" <<'PY'
import csv, sys
from collections import Counter
rows = list(csv.DictReader(open(sys.argv[1])))
train, val = [], Counter()
for r in rows:
    if r["split_partition_name"] == "train" and r["scene_name"] not in train:
        train.append(r["scene_name"])
    if r["split_partition_name"] == "val":
        val[r["scene_name"]] += 1
vals = [s for s, _ in sorted(val.items(), key=lambda kv: kv[1])[: int(sys.argv[3])]]
print(" ".join(train[: int(sys.argv[2])] + vals))
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
