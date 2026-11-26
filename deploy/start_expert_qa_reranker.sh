#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QA_DIR="$ROOT_DIR/services/expert_qa"

export CUDA_VISIBLE_DEVICES="${ATNP_RERANK_GPU:-0}"
export ATNP_RERANK_MODEL_PATH="${ATNP_RERANK_MODEL_PATH:?Set ATNP_RERANK_MODEL_PATH to the reranker model directory}"

cd "$QA_DIR"
exec "${ATNP_RERANK_PYTHON:-python}" -u -m app.reranker_server
