#!/bin/bash
# The box boots (thread H sitting 2, 2026-09-16; v3, thread V,
# 2026-09-19): the reader on the card in the background, the loader on
# 8000 in front, writing a heartbeat onto the volume from its first
# second and taking core/ off the volume (BOX_ROOT/code/CURRENT) the
# moment it is there, then becoming box_server. The Mac is never
# required to be present. Every knob is an env on the pod, set by
# pod.py --deploy, never pasted:
#   LAYOUT_KEY     the bearer key the Mac reads off the pod's env (the
#                  loader refuses code without one)
#   BOX_QUEUE_GB   the queue's budget on the volume (a RunPod volume
#                  reports the cluster's free space, so the box cannot
#                  see its own quota): 90 on the 100 GB volume, which
#                  holds nothing but the queue now
#   BOX_JOBS       tickets at once (3: one book's reading overlaps the
#                  next one's words and map)
#   BOX_IDLE_STOP  minutes with nothing to do before the box stops its
#                  pod (20): the loader's "no code on the volume and an
#                  empty queue", box_server's "nothing running, queued
#                  or asked"
#   BOX_MAX_HOURS  the ceiling since boot (36): the card's budget
#   BOX_MODEL      the reader (Qwen/Qwen2.5-VL-7B-Instruct)
#   BOX_READER_SEQS, BOX_READER_JOBS, BOX_MAP_BATCH
#                  the card's knobs, detected below; set to override
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
echo "box image $(cat /box/VERSION 2>/dev/null || echo dev) | pod ${RUNPOD_POD_ID:-none} | key ${LAYOUT_KEY:+set}${LAYOUT_KEY:-UNSET (set LAYOUT_KEY on the pod)} | queue root ${BOX_ROOT:-the container disk} | budget ${BOX_QUEUE_GB:-unset} GB | jobs ${BOX_JOBS:-2} | idle stop ${BOX_IDLE_STOP:-20} min | ceiling ${BOX_MAX_HOURS:-36} h"
# the heartbeat starts with the loader, seconds from here; the code
# comes off the volume (pod.py --deploy and --code put it there)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no card visible"
df -h / | tail -1
# the reader: localhost only (the box reads its own pages; nothing on
# the internet reaches the model). The flags are box-setup.sh's, kept:
# fp16; 16k context (a folio page at 200 dpi is about 7,800 tokens of
# image; at 8k a reading was cut off); 0.88 of the card (0.80 left
# 0.36 GiB of working memory, under one page); 6 pages in flight; the
# built-in sampler (nothing compiled at start). The map maker takes
# the rest of the card in batches of 8 (box_server's BOX_MAP_BATCH).
# THE CARD SETS THE KNOBS (2026-09-16, A: "for a rental there should be
# detection and optimization based on what it has access to"). The
# reader's weights are 15.7 GB whatever the card; everything above
# that is room for pages in flight, and a 7B model's throughput grows
# with how many it holds at once (decoding is memory-bound: more
# pages per pass, same pass). So the pages in flight (vLLM's
# max-num-seqs, and box_server's BOX_READER_JOBS to match) and the map
# maker's batch (BOX_MAP_BATCH) follow the card's memory: a 24 GB
# card 6 / 8 (the nights before), 32 to 48 GB 16 / 16, 80 GB and up
# 32 / 32 (a step, not the ceiling: measured first, raised after).
# Any of the three set on the pod's env wins over the detection.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_MB=${VRAM_MB:-24000}
if [ "$VRAM_MB" -ge 70000 ]; then TIER="80 GB and up"; SEQS=32; MAPB=32
elif [ "$VRAM_MB" -ge 30000 ]; then TIER="32 to 48 GB"; SEQS=16; MAPB=16
else TIER="24 GB"; SEQS=6; MAPB=8; fi
export BOX_READER_SEQS=${BOX_READER_SEQS:-$SEQS}
export BOX_READER_JOBS=${BOX_READER_JOBS:-$BOX_READER_SEQS}
export BOX_MAP_BATCH=${BOX_MAP_BATCH:-$MAPB}
echo "card: $VRAM_MB MiB ($TIER): $BOX_READER_SEQS pages in flight at the reader, box_server asks $BOX_READER_JOBS at once, the map maker's batch $BOX_MAP_BATCH"
echo "starting the reader ($MODEL; 15 GB fetched to the container disk, then a minute to load; the log is $BOX_READER_LOG)"
vllm serve "$MODEL" --host 127.0.0.1 --port 8001 --dtype half \
  --max-model-len 16384 --gpu-memory-utilization 0.88 \
  --max-num-seqs "$BOX_READER_SEQS" \
  --limit-mm-per-prompt '{"image": 1}' --served-model-name "$MODEL" \
  > "$BOX_READER_LOG" 2>&1 &
exec python3 /box/loader.py --port 8000
