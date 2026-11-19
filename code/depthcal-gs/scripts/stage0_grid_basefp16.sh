#!/usr/bin/env bash
# Stage-0 re-run with the frozen base weights stored in fp16 (--base-fp16): runs scripts/stage0_sizing.py once per setting, each in a fresh process.
#
#   scripts/stage0_grid_basefp16.sh <sdxl_unet_config.json> <out_dir>
#
# Writes <out_dir>/results.jsonl (one JSON line per setting; a failed run gets {"error": ...}) and
# <out_dir>/logs/<tag>.log. Alongside the harness's allocator peak it records the device-level peak seen by
# nvidia-smi (sampled every 200 ms, minus the idle baseline), which includes the CUDA context and allocator cache.
# Assumes nothing else allocates on the GPU while it runs: check nvidia-smi first.
set -u
CFG=${1:?config}
OUT=${2:?out_dir}
PY=${PY:-python}
mkdir -p "$OUT/logs"
here=$(cd "$(dirname "$0")" && pwd)

run() {  # args passed to stage0_sizing.py
  local tag
  tag=$(echo "$*" | tr -d '-' | tr ' ' '_')
  local base
  base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$OUT/logs/$tag.smi" &
  local smi=$!
  "$PY" "$here/stage0_sizing.py" --config "$CFG" "$@" > "$OUT/logs/$tag.log" 2>&1
  local rc=$?
  kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
  local peak
  peak=$(sort -n "$OUT/logs/$tag.smi" | tail -1)
  local smi_gb
  smi_gb=$(python3 -c "print(round(($peak - $base) * 1048576 / 1e9, 3))")
  local line
  line=$(grep '^{' "$OUT/logs/$tag.log" | tail -1)
  if [ $rc -eq 0 ] && [ -n "$line" ]; then
    python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['smi_peak_GB']=float(sys.argv[2]); print(json.dumps(d))" "$line" "$smi_gb" >> "$OUT/results.jsonl"
  else
    err=$(grep -m1 -o -E 'OutOfMemoryError|CUDA out of memory|non-finite loss|Error[^:]*' "$OUT/logs/$tag.log" | head -1)
    python3 -c "import json,sys; print(json.dumps({'args': sys.argv[1], 'error': sys.argv[2] or 'rc=' + sys.argv[3], 'smi_peak_GB': float(sys.argv[4])}))" "$*" "$err" "$rc" "$smi_gb" >> "$OUT/results.jsonl"
  fi
  echo "done: $* (rc=$rc)"
}

for res in 512 768; do
  for batch in 1 2; do
    run --res "$res" --batch "$batch" --rank 32 --ckpt --base-fp16
    run --res "$res" --batch "$batch" --rank 32 --base-fp16
  done
done
run --res 768 --batch 2 --rank 64 --ckpt --base-fp16
