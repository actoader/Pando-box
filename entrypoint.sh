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
#   BOX_IDLE_STOP  minutes with nothing to do before the box stops its
#                  pod (20): the loader's "no code on the volume and an
#                  empty queue", box_server's "nothing running, queued
#                  or asked"
#   BOX_MAX_HOURS  the ceiling since boot (36): the card's budget
#   BOX_MODEL      the reader (Qwen/Qwen2.5-VL-7B-Instruct), started
#                  by box_server with settings by the card (v6)
#   BOX_MAP_BATCH  the map maker's batch, detected below; set to override
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
echo "box image $(cat /box/VERSION 2>/dev/null || echo dev) | pod ${RUNPOD_POD_ID:-none} | key $([ -n "${LAYOUT_KEY:-}" ] && echo set || echo 'UNSET (set LAYOUT_KEY on the pod)') | queue root ${BOX_ROOT:-the container disk} | budget ${BOX_QUEUE_GB:-unset} GB | idle stop ${BOX_IDLE_STOP:-20} min | ceiling ${BOX_MAX_HOURS:-36} h"
# the heartbeat starts with the loader, seconds from here; the code
# comes off the volume (pod.py --deploy and --code put it there)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no card visible"
df -h / | tail -1
# THE READER IS BOX_SERVER'S (v6, 2f-d, 2026-09-28). The image starts
# no reader any more: box_server (the code the Mac sends) starts vLLM
# itself, localhost only, with settings by the card (core/dies/
# box_server.py TIERS: a page's image capped in tokens, the pages in
# flight, the share of the card; a reader that dies at start for
# memory is started again one tier down). Until v6 is built, the v5
# image's reader runs with a big card's settings (16k context, 0.88 of
# the card, 6/16/32 pages in flight by the card) and a 24 GB card
# answers by dying once, which box_server takes over from. The map
# maker's batch still follows the card here (BOX_MAP_BATCH; set on the
# pod's env to override).
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_MB=${VRAM_MB:-24000}
# a slice of a card does not say its memory ("[Insufficient Permissions]", 2026-11-16): not a number, so the least is taken
case "$VRAM_MB" in ''|*[!0-9]*) VRAM_MB=24000 ;; esac
if [ "$VRAM_MB" -ge 70000 ]; then TIER="80 GB and up"; MAPB=32
elif [ "$VRAM_MB" -ge 30000 ]; then TIER="32 to 48 GB"; MAPB=16
else TIER="24 GB"; MAPB=8; fi
export BOX_MAP_BATCH=${BOX_MAP_BATCH:-$MAPB}
export BOX_MODEL="$MODEL"
echo "card: $VRAM_MB MiB ($TIER): the map maker's batch $BOX_MAP_BATCH; the reader ($MODEL) is started by box_server for this card (its log is $BOX_READER_LOG)"
# the loader off the volume when the Mac has put one there (pod.py
# put_code: box/code/loader.py beside the code), the image's own as
# the fallback and when the volume's cannot even start: a loader
# change is a --code, never an image build (thread V, sitting 2)
if [ -n "${BOX_ROOT:-}" ] && [ -s "$BOX_ROOT/code/loader.py" ]; then
  echo "loader: the volume's ($BOX_ROOT/code/loader.py)"
  python3 "$BOX_ROOT/code/loader.py" --port 8000 && exit 0
  echo "the volume's loader ended; falling back to the image's"
fi
exec python3 /box/loader.py --port 8000
