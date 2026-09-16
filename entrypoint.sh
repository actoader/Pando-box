#!/bin/bash
# The box boots (thread H sitting 2, 2026-09-16): the reader on the
# card in the background, the loader on 8000 in front, until the Mac
# sends core/ and the loader becomes box_server. Every knob is an env
# on the pod, set once on its template, never pasted:
#   LAYOUT_KEY     the bearer key the Mac reads off the pod's env (the
#                  loader refuses code without one)
#   BOX_QUEUE_GB   the queue's budget on the volume (a RunPod volume
#                  reports the cluster's free space, so the box cannot
#                  see its own quota): 90 on the 100 GB volume, which
#                  holds nothing but the queue now
#   BOX_JOBS       tickets at once (3: one book's reading overlaps the
#                  next one's words and map)
#   BOX_IDLE_STOP  minutes idle before the box stops its pod (20)
#   BOX_MODEL      the reader (Qwen/Qwen2.5-VL-7B-Instruct)
set -u
MODEL=${BOX_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}   # the container disk
export BOX_READER_LOG=/root/vllm.log
if [ -d /workspace ] && [ -w /workspace ]; then
  export BOX_ROOT=${BOX_ROOT:-/workspace/box}
  # the pet machine's leftovers on the volume (the venv, the weights,
  # the pasted code and key of the nights before the image): swept
  # once, so the volume holds the queue and nothing else
  for f in .venv-box .venv-layout hf llm core LAYOUT_KEY pando-core.tgz box-setup.sh vllm.log box_server.log watchdog.log; do
    if [ -e "/workspace/$f" ]; then echo "sweeping /workspace/$f (the hand-built box's; the image carries it now)"; rm -rf "/workspace/$f"; fi
  done
fi
echo "box image $(cat /box/VERSION 2>/dev/null || echo dev) | pod ${RUNPOD_POD_ID:-none} | key ${LAYOUT_KEY:+set}${LAYOUT_KEY:-UNSET (set LAYOUT_KEY on the pod)} | queue root ${BOX_ROOT:-the container disk} | budget ${BOX_QUEUE_GB:-unset} GB | jobs ${BOX_JOBS:-2}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no card visible"
df -h / | tail -1
# the reader: localhost only (the box reads its own pages; nothing on
# the internet reaches the model). The flags are box-setup.sh's, kept:
# fp16; 16k context (a folio page at 200 dpi is about 7,800 tokens of
# image; at 8k a reading was cut off); 0.88 of the card (0.80 left
# 0.36 GiB of working memory, under one page); 6 pages in flight; the
# built-in sampler (nothing compiled at start). The map maker takes
# the rest of the card in batches of 8 (box_server's BOX_MAP_BATCH).
echo "starting the reader ($MODEL; 15 GB fetched to the container disk, then a minute to load; the log is $BOX_READER_LOG)"
vllm serve "$MODEL" --host 127.0.0.1 --port 8001 --dtype half \
  --max-model-len 16384 --gpu-memory-utilization 0.88 --max-num-seqs 6 \
  --limit-mm-per-prompt '{"image": 1}' --served-model-name "$MODEL" \
  > "$BOX_READER_LOG" 2>&1 &
exec python3 /box/loader.py --port 8000
