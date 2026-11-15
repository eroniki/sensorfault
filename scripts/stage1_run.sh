#!/usr/bin/env bash
# Launch the full stage-1 run detached (tmux if available, else nohup), resuming from the newest checkpoint in $STAGE1_WORK.
#   export SDXL_DIR=... BLENDEDMVS_ROOT=... HYPERSIM_ROOT=... HYPERSIM_SPLIT_CSV=...  STAGE1_WORK=runs/stage1
#   scripts/stage1_run.sh            start or resume
# Stop cleanly (checkpoint, then exit): kill -TERM $(cat $STAGE1_WORK/pid)
# One heavy job at a time: refuses to start with less than 16 GB free on the GPU (mapgend or another run is using it); FORCE=1 overrides.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
: "${SDXL_DIR:?}"; : "${STAGE1_WORK:?}"
mkdir -p "$STAGE1_WORK"
if [ -f "$STAGE1_WORK/pid" ] && kill -0 "$(cat "$STAGE1_WORK/pid")" 2>/dev/null; then echo "already running (pid $(cat "$STAGE1_WORK/pid"))"; exit 0; fi
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${FORCE:-0}" != 1 ] && [ "$free" -lt 16000 ]; then echo "only $free MiB free on the GPU: another heavy job is running (FORCE=1 to override)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 3; fi
cmd="cd $here && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && PYTHONPATH=src ${PY:-python} scripts/stage1_train.py --config ${CONFIG:-configs/stage1_sdxl_lora.json} --workdir $STAGE1_WORK --resume auto >> $STAGE1_WORK/console.log 2>&1 & echo \$! > $STAGE1_WORK/pid; wait"
if command -v tmux >/dev/null; then
  tmux new-session -d -s "${TMUX_NAME:-depthcal-stage1}" "bash -lc '$cmd'"
  echo "started in tmux session ${TMUX_NAME:-depthcal-stage1}"
else
  nohup bash -lc "$cmd" >/dev/null 2>&1 &
  echo "started with nohup"
fi
sleep 5
echo "pid: $(cat "$STAGE1_WORK/pid" 2>/dev/null || echo unknown); log: $STAGE1_WORK/console.log ; metrics: $STAGE1_WORK/log.jsonl"
