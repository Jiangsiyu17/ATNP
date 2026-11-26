"""Optional HTTP cross-encoder scoring with validated responses and rule fallback.

Raw logits and reciprocal-rank fusion are relevance measures, not probabilities
of factual correctness. Scientific attribution/verification remains separate.
"""
import math
import os
import time

import requests

MODEL_ID = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
MAX_CANDIDATES = 80


def backend_name(value=None):
    backend = value or os.getenv('ATNP_RERANK_BACKEND', 'rules')
    if backend not in {'rules', 'neural', 'hybrid'}:
        raise ValueError('rerank_backend must be rules, neural or hybrid')
    return backend


def service_status():
    mode = backend_name()
    if mode == 'rules':
        return {'status': 'disabled', 'backend': mode}
    try:
        url = os.getenv('ATNP_RERANK_URL', 'http://127.0.0.1:18081').rstrip('/')
        response = requests.get(url + '/health', timeout=(1, 2))
        response.raise_for_status()
        data = response.json()
        if data.get('model') != MODEL_ID or data.get('revision') != REVISION:
            raise ValueError('Unexpected reranker model revision')
        return {**data, 'backend': mode}
    except (requests.RequestException, ValueError, AttributeError) as exc:
        return {'status': 'unavailable', 'backend': mode, 'error_type': type(exc).__name__}


def rerank_rows(question, ranked, *, backend=None, fuse_rules=True):
    """Input rows contain item, rerank_score, original_index; preserve row provenance.

    Cheap rule ranking supplies the top-80 pool. No unscored rows receive neural
    scores. On any error the *entire* original ordering is returned unchanged.
    """
    mode = backend_name(backend)
    info = {'requested_backend': mode, 'effective_backend': 'rules',
            'status': 'disabled' if mode == 'rules' else 'no_candidates',
            'candidate_count': len(ranked), 'scored_count': 0, 'elapsed_ms': 0}
    if mode == 'rules' or not ranked:
        return ranked, info
    started = time.monotonic()
    pool = ranked[:MAX_CANDIDATES]
    documents = [str(row['item'].get('evidence_sentence') or '')[:4200] + '\nTitle: '
                 + str(row['item'].get('title') or '')[:300] for row in pool]
    try:
        url = os.getenv('ATNP_RERANK_URL', 'http://127.0.0.1:18081').rstrip('/')
        response = requests.post(url + '/v1/rerank', json={'query': question[:2000], 'documents': documents},
                                 timeout=(2, 15))
        response.raise_for_status()
        data = response.json()
        if data.get('model') != MODEL_ID or data.get('revision') != REVISION:
            raise ValueError('Unexpected reranker model revision')
        results = data.get('results')
        if not isinstance(results, list) or len(results) != len(pool):
            raise ValueError('Missing reranker scores')
        scores = {}
        for row in results:
            index, score = row.get('index'), row.get('score')
            if (type(index) is not int or index not in range(len(pool)) or index in scores
                    or type(score) not in (int, float) or not math.isfinite(score)):
                raise ValueError('Invalid reranker score or index')
            scores[index] = float(score)
        neural_order = sorted(scores, key=lambda i: (-scores[i], i))
        neural_ranks = {i: rank for rank, i in enumerate(neural_order, 1)}
        hybrid = mode == 'hybrid' and fuse_rules
        output = []
        for i, row in enumerate(pool):
            # Rank fusion avoids adding logits to incomparable heuristic scores.
            score = ((0.7 / (60 + neural_ranks[i]) + 0.3 / (60 + i + 1)) if hybrid
                     else 1.0 / (60 + neural_ranks[i]))
            output.append({**row, 'rule_score': row['rerank_score'], 'neural_score': scores[i],
                           'neural_rank': neural_ranks[i], 'rerank_score': score})
        maximum = max(r['rerank_score'] for r in output)
        for row in output:
            row['rerank_score'] = round(row['rerank_score'] / maximum, 6)
        output.sort(key=lambda r: (-r['rerank_score'], r['original_index']))
        info.update(status='applied', effective_backend='hybrid' if hybrid else 'neural',
                    model=data['model'], revision=data['revision'], scored_count=len(pool),
                    truncated_count=len(ranked)-len(pool), max_length=data.get('max_length'),
                    score_kind='relative_rrf' if hybrid else 'relative_neural_rank',
                    server_elapsed_ms=data.get('elapsed_ms'))
        return output, info
    except (requests.RequestException, ValueError, TypeError, KeyError, AttributeError) as exc:
        info.update(status='fallback', error_type=type(exc).__name__)
        return ranked, info
    finally:
        info['elapsed_ms'] = round((time.monotonic()-started)*1000)
