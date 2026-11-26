#!/usr/bin/env bash
set -euo pipefail

: "${ATNP_MODEL_PATH:?Set ATNP_MODEL_PATH to the local Qwen model directory}"

export CUDA_VISIBLE_DEVICES="${ATNP_MODEL_GPUS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

exec "${ATNP_VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
  --model "$ATNP_MODEL_PATH" \
  --served-model-name "${VLLM_MODEL:-qwen32b-atnp}" \
  --host 127.0.0.1 \
  --port "${ATNP_MODEL_PORT:-18000}" \
  --dtype "${ATNP_MODEL_DTYPE:-bfloat16}" \
  --tensor-parallel-size "${ATNP_TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${ATNP_GPU_MEMORY_UTILIZATION:-0.88}" \
  --max-model-len "${ATNP_MAX_MODEL_LEN:-12288}" \
  --max-num-seqs "${ATNP_MAX_NUM_SEQS:-2}" \
  --enable-chunked-prefill \
  --max-num-batched-tokens "${ATNP_MAX_BATCHED_TOKENS:-2048}" \
  --disable-log-requests
