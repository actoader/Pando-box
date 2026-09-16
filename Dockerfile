# The box as an image: the MACHINE only (thread H sitting 2, 2026-09-16).
#
# WHY an image: every box night before this one was lost to assembling
# the machine by hand on a volume (tools wiped each Stop, a venv wanting
# ninja, a resize that emptied the weights). PRINCIPLES.md rule 12: the
# right shape was an image and a job API from the start.
# WHY no Pando code in it: core/ changes every sitting; an image with the
# code inside is rebuilt and pushed before every night. So the image
# boots box/loader.py, the Mac sends core/ after Start, and the loader
# becomes box_server. This file changes when vLLM or a language pack
# does, a few times a year, never for a sitting's code.
#
# Built by GitHub Actions from the pando-box repo (box/build.yml is the
# workflow) and published as ghcr.io/<owner>/pando-box:<tag>.
#
# FROM: vLLM's own image, pinned (a floating tag changes under us):
# CUDA, torch, transformers, vllm 0.29.0, Pillow, huggingface_hub, the
# vllm CLI and python3 on PATH. Ubuntu underneath, so apt.
FROM vllm/vllm-openai:v0.29.0

ARG VERSION=dev
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PANDO_LAYOUT_MODEL=/pando/llm/layout-heron \
    VLLM_USE_FLASHINFER_SAMPLER=0

# the tools: poppler renders pages, tesseract reads them (eng comes
# with tesseract-ocr; nine packs beside it, the shelf's languages),
# wamerican is the words list complete.hear_language reads
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      poppler-utils tesseract-ocr tesseract-ocr-deu tesseract-ocr-fra \
      tesseract-ocr-ell tesseract-ocr-lat tesseract-ocr-ita \
      tesseract-ocr-spa tesseract-ocr-nld tesseract-ocr-por \
      tesseract-ocr-rus wamerican curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && tesseract --list-langs

# the meter's second arm: runpodctl (the first is RunPod's API v2 with
# the pod-scoped key RunPod sets in every pod)
RUN curl -sSL https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64 \
      -o /usr/local/bin/runpodctl && chmod +x /usr/local/bin/runpodctl

# the layout weights (docling-project/docling-layout-heron, Apache-2.0,
# 170 MB): baked in, so the map maker never fetches at 3 am. The three
# files layout_worker.find_model needs, nothing else.
RUN mkdir -p /pando/llm/layout-heron && cd /pando/llm/layout-heron \
    && for f in config.json preprocessor_config.json model.safetensors; do \
         curl -sSL "https://huggingface.co/docling-project/docling-layout-heron/resolve/main/$f" -o "$f"; \
       done \
    && ls -la && test -s model.safetensors

# the reader's weights (Qwen2.5-VL-7B-Instruct, 15 GB) are NOT here:
# fetched at boot to the container disk at datacenter speed (3 to 5
# minutes) by vllm serve. Never on the volume: a resize damaged one.

COPY entrypoint.sh loader.py /box/
RUN chmod +x /box/entrypoint.sh && echo "$VERSION" > /box/VERSION \
    && python3 -c "import vllm, torch, transformers, PIL; print('vllm', vllm.__version__, 'torch', torch.__version__)"

EXPOSE 8000
ENTRYPOINT ["/box/entrypoint.sh"]
