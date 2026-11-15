#!/usr/bin/env bash
# Stage-1 smoke run on real weights: 200 steps in two legs (stop at 100, then resume), then the verdict.
#   export SDXL_DIR=... BLENDEDMVS_ROOT=... [HYPERSIM_ROOT=... HYPERSIM_SPLIT_CSV=...]   STAGE1_WORK=runs/smoke
#   scripts/stage1_smoke.sh [extra --set args, e.g. datasets='[{"type":"blendedmvs","root":"...","weight":1}]']
# One heavy job at a time: refuses to start if the GPU already has less than 14 GB free (set FORCE=1 to override).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
: "${SDXL_DIR:?}"; : "${STAGE1_WORK:?}"
mkdir -p "$STAGE1_WORK"
if command -v nvidia-smi >/dev/null; then
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${FORCE:-0}" != 1 ] && [ "$free" -lt 14000 ]; then echo "only $free MiB free on the GPU: another job is running (set FORCE=1 to override)"; exit 3; fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$here"
py=${PY:-python}
echo "=== leg 1: steps 1-100"
PYTHONPATH=src $py scripts/stage1_train.py --config configs/stage1_smoke.json --workdir "$STAGE1_WORK" --resume none --stop-at 100 --set "$@" 2>&1 | tee "$STAGE1_WORK/console_leg1.log"
echo "=== leg 2: resume to step 200"
PYTHONPATH=src $py scripts/stage1_train.py --config configs/stage1_smoke.json --workdir "$STAGE1_WORK" --resume auto --set "$@" 2>&1 | tee "$STAGE1_WORK/console_leg2.log"
$py scripts/stage1_smoke_check.py "$STAGE1_WORK" "$STAGE1_WORK/console_leg2.log" | tee "$STAGE1_WORK/smoke_verdict.txt"
