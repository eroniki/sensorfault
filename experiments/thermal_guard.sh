#!/bin/bash
# Run a command under a temperature guard: SIGSTOP it while the GPU or CPU (Tctl) is above LIMIT (80 C),
# SIGCONT once both are below RESUME (72 C). Logs max temperatures to $GUARD_LOG (default /dev/stderr).
# Usage: experiments/thermal_guard.sh <command ...>
LIMIT=${LIMIT:-80}; RESUME=${RESUME:-72}; LOG=${GUARD_LOG:-/dev/stderr}
"$@" & pid=$!
maxg=0; maxc=0; paused=0
temps() {
  g=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  c=$(sensors 2>/dev/null | sed -n 's/.*Tctl:[^0-9]*\([0-9.]*\).*/\1/p' | head -1)
  g=${g:-0}; c=${c:-0}
}
while kill -0 $pid 2>/dev/null; do
  temps
  maxg=$(echo "$g $maxg" | awk '{print ($1>$2)?$1:$2}'); maxc=$(echo "$c $maxc" | awk '{print ($1>$2)?$1:$2}')
  hot=$(echo "$g $c $LIMIT" | awk '{print ($1>$3||$2>$3)?1:0}')
  cool=$(echo "$g $c $RESUME" | awk '{print ($1<$3&&$2<$3)?1:0}')
  if [ $paused = 0 ] && [ $hot = 1 ]; then kill -STOP $pid; paused=1; echo "guard: paused at GPU $g C, CPU $c C" >> $LOG; fi
  if [ $paused = 1 ] && [ $cool = 1 ]; then kill -CONT $pid; paused=0; echo "guard: resumed at GPU $g C, CPU $c C" >> $LOG; fi
  sleep 5
done
wait $pid; rc=$?
echo "guard: done rc=$rc max GPU ${maxg} C, max CPU ${maxc} C" >> $LOG
exit $rc
