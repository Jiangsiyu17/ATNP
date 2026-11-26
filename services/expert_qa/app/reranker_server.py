"""Local, bounded cross-encoder service; isolated from the generation model."""
import math
import json
import os
from pathlib import Path
import threading
import time

from flask import Flask, jsonify, request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
MODEL_PATH = os.getenv('ATNP_RERANK_MODEL_PATH', '/tmp/atnp-reranker-models/bge-reranker-v2-m3')
DEVICE = os.getenv('ATNP_RERANK_DEVICE', 'cuda:0')
BATCH_SIZE = 8
MAX_LENGTH = 512
MAX_DOCUMENTS = 80
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 3_000_000
_lock = threading.Lock()
_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return
    if not Path(MODEL_PATH, 'model.safetensors').is_file():
        raise RuntimeError('Reranker weights are not installed')
    manifest = json.loads(Path(MODEL_PATH, 'atnp_model_manifest.json').read_text())
    if manifest.get('model') != MODEL_ID or manifest.get('revision') != REVISION:
        raise RuntimeError('Reranker installation does not match the pinned model revision')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float16 if DEVICE.startswith('cuda') else torch.float32)
    if model.config.num_labels != 1:
        raise RuntimeError('Expected a single-logit relevance model')
    model.to(DEVICE).eval()
    _tokenizer, _model = tokenizer, model


@app.get('/health')
def health():
    return jsonify(status='ok' if _model is not None else 'not_loaded',
                   model=MODEL_ID, revision=REVISION, device=DEVICE,
                   max_documents=MAX_DOCUMENTS, max_length=MAX_LENGTH)


@app.post('/v1/rerank')
def rerank():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error='JSON object required'), 400
    query, documents = payload.get('query'), payload.get('documents')
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        return jsonify(error='query must contain 1..2000 characters'), 400
    if (not isinstance(documents, list) or not 1 <= len(documents) <= MAX_DOCUMENTS
            or any(not isinstance(d, str) or not d.strip() or len(d) > 5000 for d in documents)):
        return jsonify(error='documents must contain 1..80 nonempty texts, at most 5000 characters each'), 400
    # No unbounded queue of GPU requests. The client can fall back to rules.
    if not _lock.acquire(timeout=1):
        return jsonify(error='reranker_busy'), 503
    started = time.monotonic()
    try:
        if _model is None:
            return jsonify(error='reranker_not_loaded'), 503
        # Cap the query before pair encoding so long questions cannot consume the passage budget.
        query_ids = _tokenizer.encode(query, add_special_tokens=False, truncation=True, max_length=128)
        bounded_query = _tokenizer.decode(query_ids, skip_special_tokens=True)
        scores = []
        with torch.inference_mode():
            for start in range(0, len(documents), BATCH_SIZE):
                batch = documents[start:start + BATCH_SIZE]
                inputs = _tokenizer([bounded_query] * len(batch), batch, padding=True,
                                    truncation='only_second', max_length=MAX_LENGTH,
                                    return_tensors='pt').to(DEVICE)
                scores.extend(_model(**inputs).logits.reshape(-1).float().cpu().tolist())
        if len(scores) != len(documents) or not all(math.isfinite(s) for s in scores):
            raise RuntimeError('Invalid model scores')
        return jsonify(model=MODEL_ID, revision=REVISION, max_length=MAX_LENGTH,
                       elapsed_ms=round((time.monotonic()-started)*1000),
                       results=[{'index': i, 'score': score} for i, score in enumerate(scores)])
    except Exception:
        app.logger.exception('Reranker inference failed')
        if DEVICE.startswith('cuda'):
            torch.cuda.empty_cache()
        return jsonify(error='reranker_inference_failed'), 503
    finally:
        _lock.release()


if __name__ == '__main__':
    torch.set_num_threads(4)
    with _lock:
        load_model()
    app.run(host='127.0.0.1', port=int(os.getenv('ATNP_RERANK_PORT', '18081')), threaded=True, use_reloader=False)
