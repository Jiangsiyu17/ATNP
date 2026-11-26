#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QA_DIR="$ROOT_DIR/services/expert_qa"

export ATNP_QA_DATA_DIR="${ATNP_QA_DATA_DIR:-$QA_DIR/data}"
export ATNP_KG_DB="${ATNP_KG_DB:-$ROOT_DIR/crawler/knowledge_base_full.duckdb}"
export ATNP_PASSAGE_INDEX="${ATNP_PASSAGE_INDEX:-$ROOT_DIR/crawler/passages.sqlite}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:18000}"
export VLLM_MODEL="${VLLM_MODEL:-qwen32b-atnp}"
export ATNP_MODEL_BACKEND="${ATNP_MODEL_BACKEND:-vllm}"
export ATNP_QUEUE_MAX_RUNNING="${ATNP_QUEUE_MAX_RUNNING:-2}"
export ATNP_QUEUE_MAX_WAITING="${ATNP_QUEUE_MAX_WAITING:-4}"
export ATNP_QUEUE_WAIT_TIMEOUT="${ATNP_QUEUE_WAIT_TIMEOUT:-0}"
export ATNP_RERANK_BACKEND="${ATNP_RERANK_BACKEND:-hybrid}"
export ATNP_RERANK_URL="${ATNP_RERANK_URL:-http://127.0.0.1:18081}"
export ATNP_PERFORMANCE_LOG_PATH="${ATNP_PERFORMANCE_LOG_PATH:-$ROOT_DIR/logs/expert_qa_performance.jsonl}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$ROOT_DIR/logs"
cd "$QA_DIR"
exec "${ATNP_QA_PYTHON:-python}" -m uvicorn app.api:app \
  --host "${ATNP_QA_HOST:-127.0.0.1}" \
  --port "${ATNP_QA_PORT:-8080}"
