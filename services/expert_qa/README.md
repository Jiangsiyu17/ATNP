# ATNP Expert Q&A service

This directory contains only the production FastAPI retrieval service. The
one-shot Expert Q&A page remains in the main Django `web` application.

The service reads, but never builds or modifies, the shared retrieval data:

- `ATNP_KG_DB`: knowledge-graph DuckDB file;
- `ATNP_PASSAGE_INDEX`: local SQLite FTS5 passage index;
- `VLLM_BASE_URL`: OpenAI-compatible local model endpoint.

Large databases, model weights, logs, evaluation runs and development reports
must remain outside Git. See `deploy/EXPERT_QA_DEPLOY_ZH.md` for deployment.
