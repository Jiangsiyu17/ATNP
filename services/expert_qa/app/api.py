from app.performance import request_identity
import asyncio
import json
import logging
import os
import hashlib
import time
import uuid
from functools import partial
from typing import Any

import duckdb
import requests
from fastapi import FastAPI, HTTPException, Query, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from app.literature_tools import passage_index_status
from app.neural_reranker import service_status as reranker_service_status
from typing import Literal
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)
from starlette.concurrency import run_in_threadpool

from app.kg_retriever import DB_PATH
from app.model_client import VLLM_MODEL, check_model_server
# 整个文件真正的问题逻辑
from app.qa_pipeline import run_qa
from app.question_parser import suggest_compounds
from app.request_queue import RequestQueue, AdmissionError, RequestCancelled, cancellation_scope


LOGGER = logging.getLogger("atnp.api")
MODEL_BACKEND = os.getenv('ATNP_MODEL_BACKEND', 'legacy')
MAX_RUNNING = int(os.getenv('ATNP_QUEUE_MAX_RUNNING', '1'))
if MAX_RUNNING > 1 and MODEL_BACKEND != 'vllm':
    raise ValueError('Concurrent QA requires the dedicated vLLM backend; legacy generation is serial')
REQUEST_QUEUE = RequestQueue(
    max_waiting=int(os.getenv('ATNP_QUEUE_MAX_WAITING', '4')),
    wait_timeout=float(os.getenv('ATNP_QUEUE_WAIT_TIMEOUT', '0')),
    max_running=MAX_RUNNING,
)
_BACKGROUND_WORKERS = set()
STREAM_HEARTBEAT_SECONDS = 15
API_VERSION = "0.10.0"
def _cors_origins():
    value = os.getenv(
        "ATNP_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000"
    )
    return [
        origin.strip()
        for origin in value.split(",")
        if origin.strip()
    ]


app = FastAPI(
    title="ATNP Expert Q&A API",
    description=(
        "Evidence-grounded natural-product question answering "
        "with structured literature citations."
    ),
    version=API_VERSION
)
# 前后端通常运行在不同端口甚至不同域名下，浏览器会受到同源策略限制，所以后端通过CORS白名单允许指定前端访问接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="Natural-language question in Chinese or English"
    )
    compound: str | None = Field(
        default=None,
        max_length=200,
        description="Optional canonical compound name override"
    )
    context_compound: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Compound inherited from the current conversation when the "
            "new question does not name one explicitly"
        )
    )
    model: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Model identifier returned by GET /api/models; omitted uses the "
            "configured default model"
        )
    )
    conversation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Stable client-managed conversation identifier"
    )
    validate_claims: bool | None = Field(
        default=None,
        description=(
            "true: always validate; false: disable; null/omitted: "
            "automatically validate mechanism and biological-effect questions"
        )
    )
    # Compatibility field: online retrieval is restricted to local data.
    external_search: Literal[False] = Field(default=False, deprecated=True,
        description="Local-only retrieval. This legacy field only accepts false; external search has been removed.")
    # 任务路由
    task_router: bool = Field(default=True, description="Enable LLM domain, retrieval and intent routing")

    search_agent: bool = Field(
        default=True,
        description="Enable bounded model-directed KG/text retrieval; false uses the baseline workflow"
    )
    include_debug: bool = Field(
        default=False,
        description="Include candidates, evidence and the complete prompt"
    )
    edge_limit: int = Field(
        default=100,
        ge=1,
        le=300,
        description="Maximum knowledge-graph candidate pool"
    )
    evidence_limit: int = Field(
        default=300,
        ge=1,
        le=1000,
        description="Maximum balanced evidence candidate pool"
    )
    final_evidence_limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Legacy hard cap; dynamic reranking is used by default"
    )
    rerank_min_evidence: int = Field(default=6, ge=1, le=50)
    rerank_max_evidence: int = Field(default=25, ge=1, le=100)
    rerank_token_budget: int = Field(default=7000, ge=500, le=30000)
    rerank_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    rerank_backend: Literal['rules', 'neural', 'hybrid'] | None = Field(
        default=None, description='Ranking backend; null uses server configuration. Neural failures fall back to rules.')
    max_tokens: int = Field(default=768, ge=32, le=2048)
    timeout: int = Field(default=600, ge=10, le=1800)
    claim_max_tokens: int = Field(default=2000, ge=128, le=4096)
    claim_timeout: int = Field(default=600, ge=10, le=1800)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("question cannot be blank")

        return value

    @field_validator(
        "compound",
        "context_compound",
        "conversation_id",
        "model"
    )
    @classmethod
    def strip_optional_text(cls, value):
        if value is None:
            return None

        value = value.strip()
        return value or None

    @model_validator(mode="after")
    def validate_rerank_bounds(self):
        if self.rerank_min_evidence > self.rerank_max_evidence:
            raise ValueError(
                "rerank_min_evidence cannot exceed rerank_max_evidence"
            )

        return self


class ChatResponse(BaseModel):
    request_id: str
    conversation_id: str
    elapsed_ms: int
    status: str
    message: str | None = None
    question: str
    model: str
    compound: str | None = None
    parsed: dict[str, Any]
    answer: str | None = None
    references: list[dict[str, Any]]
    kg_relations: list[dict[str, Any]]
    retrieval: dict[str, Any] = Field(default_factory=dict)
    citation_validation: dict[str, Any] | None = None
    language_validation: dict[str, Any] | None = None
    claim_validation: dict[str, Any] | None = None
    answer_coverage_validation: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None


def _check_database():
    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True
    )

    try:
        counts = {}

        for table in ("nodes", "edges", "evidence"):
            counts[table] = connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]

        return {
            "status": "ok",
            "path": str(DB_PATH),
            "counts": counts
        }
    finally:
        connection.close()


def _model_items(model_data):
    items = []
    seen = set()
    for item in (model_data or {}).get("data") or []:
        model_id = str(item.get("id") or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        items.append({
            "id": model_id,
            "label": str(item.get("display_name") or model_id),
            "owned_by": str(item.get("owned_by") or "unknown")
        })
    return items


def _resolve_requested_model(requested_model=None):
    selected = str(requested_model or VLLM_MODEL).strip()
    if not requested_model:
        return selected

    model_data = check_model_server(timeout=3)
    available_ids = {
        item["id"] for item in _model_items(model_data)
    }
    if selected not in available_ids:
        raise ValueError(
            f"Requested model '{selected}' is not available. "
            f"Available models: {', '.join(sorted(available_ids)) or 'none'}."
        )
    return selected


def _build_qa_call(payload, progress_callback=None, model_name=None):
    return partial(
        run_qa,
        external_search=payload.external_search,
        task_router=payload.task_router,
        search_agent=payload.search_agent,
        question=payload.question,
        compound=payload.compound,
        context_compound=payload.context_compound,
        model=model_name or payload.model or VLLM_MODEL,
        edge_limit=payload.edge_limit,
        evidence_limit=payload.evidence_limit,
        final_evidence_limit=payload.final_evidence_limit,
        rerank_min_evidence=payload.rerank_min_evidence,
        rerank_max_evidence=payload.rerank_max_evidence,
        rerank_token_budget=payload.rerank_token_budget,
        rerank_score_threshold=payload.rerank_score_threshold,
        rerank_backend=payload.rerank_backend,
        call_model=True,
        model_max_tokens=payload.max_tokens,
        model_timeout=payload.timeout,
        validate_claims=payload.validate_claims,
        claim_validation_max_tokens=payload.claim_max_tokens,
        claim_validation_timeout=payload.claim_timeout,
        progress_callback=progress_callback
    )


def _stream_error(exc, request_id):
    if isinstance(exc, requests.Timeout):
        status_code = 504
        detail = "The local model timed out while generating a response."
    elif isinstance(exc, requests.RequestException):
        status_code = 503
        detail = "The local model service is unavailable."
    elif isinstance(exc, ValueError):
        status_code = 422
        detail = str(exc)
    else:
        status_code = 500
        detail = f"Internal Q&A error ({request_id})."

    return {
        "request_id": request_id,
        "status_code": status_code,
        "detail": detail
    }


def _sse_event(event_name, payload):
    data = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":")
    )
    return f"event: {event_name}\ndata: {data}\n\n"


def _public_result(
    result,
    request_id,
    conversation_id,
    elapsed_ms,
    include_debug
):
    claim_validation = result.get("claim_validation")

    if claim_validation and not include_debug:
        claim_validation = dict(claim_validation)
        claim_validation.pop("raw_judge_response", None)

    response = {
        "request_id": request_id,
        "conversation_id": conversation_id,
        "elapsed_ms": elapsed_ms,
        "status": result["status"],
        "message": result.get("message"),
        "question": result["question"],
        "model": result.get("model") or VLLM_MODEL,
        "compound": result.get("compound"),
        "parsed": result.get("parsed") or {},
        "answer": result.get("answer"),
        "references": result.get("references") or [],
        "kg_relations": result.get("relevant_edges") or [],
        "retrieval": result.get("retrieval") or {},
        "citation_validation": result.get("citation_validation"),
        "language_validation": result.get("language_validation"),
        "claim_validation": claim_validation,
        "answer_coverage_validation": result.get(
            "answer_coverage_validation"
        ),
        "debug": None
    }

    if include_debug:
        response["debug"] = {
            "candidate_edges": result.get("candidate_edges") or [],
            "candidate_evidence": result.get("candidate_evidence") or [],
            "relevant_evidence": result.get("relevant_evidence") or [],
            "prompt": result.get("prompt")
        }

    return response


@app.get("/")
async def root():
    return {
        "name": "ATNP Expert Q&A API",
        "version": API_VERSION,
        "expert_qa": "served by the ATNP Django web application",
        "health": "/api/health",
        "docs": "/docs"
    }


@app.get("/api/health")
@app.get("/expert-qa/api/health")
async def health():
    database = await run_in_threadpool(
        _check_database
    )

    try:
        model_data = await run_in_threadpool(
            partial(check_model_server, timeout=3)
        )
        model_status = {
            "status": "ok",
            "configured_model": VLLM_MODEL,
            "response": model_data
        }
    except Exception as exc:
        model_status = {
            "status": "unavailable",
            "configured_model": VLLM_MODEL,
            "error": f"{type(exc).__name__}: {exc}"
        }

    overall_status = (
        "ok"
        if model_status["status"] == "ok"
        else "degraded"
    )

    return {
        "status": overall_status,
        "api_version": API_VERSION,
        "admission": {"model_backend": MODEL_BACKEND, "max_running": REQUEST_QUEUE.max_running,
                      "max_waiting": REQUEST_QUEUE.max_waiting,
                      "queue_timeout_seconds": REQUEST_QUEUE.wait_timeout},
        "database": database,
        "retrieval_scope": {"mode": "local_only", "external_search_enabled": False,
                            "sources": ["knowledge_graph", "evidence_text", "local_passage"]},
        "fulltext_index": await run_in_threadpool(passage_index_status),
        "reranker": await run_in_threadpool(reranker_service_status),
        "model": model_status
    }


@app.get("/api/models")
@app.get("/expert-qa/api/models")
async def available_models():
    try:
        model_data = await run_in_threadpool(
            partial(check_model_server, timeout=3)
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="The local model service is unavailable."
        ) from exc

    items = _model_items(model_data)
    if not items:
        raise HTTPException(
            status_code=503,
            detail="The model service returned no available models."
        )
    available_ids = {item["id"] for item in items}
    default_model = (
        VLLM_MODEL if VLLM_MODEL in available_ids else items[0]["id"]
    )
    return {
        "default_model": default_model,
        "items": items
    }


@app.get("/api/compounds/suggest")
@app.get("/expert-qa/api/compounds/suggest")
async def compound_suggestions(
    q: str = Query(min_length=1, max_length=100),
    limit: int = Query(default=10, ge=1, le=50)
):
    items = await run_in_threadpool(
        partial(
            suggest_compounds,
            query=q,
            limit=limit
        )
    )

    return {
        "query": q,
        "items": items
    }


def _admit(payload, request_id, conversation_id):
    config = payload.model_dump(exclude={'conversation_id', 'include_debug'})
    config['question'] = ' '.join(payload.question.split())
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    try:
        return REQUEST_QUEUE.submit(request_id, conversation_id + ':' + fingerprint)
    except AdmissionError as exc:
        raise HTTPException(exc.status_code, exc.detail, headers={'Retry-After': '5'} if exc.status_code == 429 else None) from exc


def _retain_task(task):
    _BACKGROUND_WORKERS.add(task)
    def finished(done):
        _BACKGROUND_WORKERS.discard(done)
        try:
            done.exception()
        except (asyncio.CancelledError, RequestCancelled):
            pass
    task.add_done_callback(finished)
    return task


async def _execute_job(job, qa_call, request=None, conversation_id=None):
    queue = REQUEST_QUEUE
    worker_task = None
    try:
        while not queue.try_start(job):
            state = queue.snapshot(job)['state']
            if state == 'queue_timeout':
                raise AdmissionError(408, '排队等待已超时，本次任务未执行，请稍后重试。')
            if state == 'cancelled':
                raise RequestCancelled()
            if request is not None and await request.is_disconnected():
                raise RequestCancelled()
            await asyncio.sleep(0.25)

        def worker():
            failed = True
            try:
                with cancellation_scope(job.cancelled), request_identity(job.request_id, conversation_id):
                    result = qa_call()
                failed = False
                return result
            finally:
                queue.finish(job, failed=failed)

        worker_task = _retain_task(asyncio.create_task(run_in_threadpool(worker)))
        while not worker_task.done():
            if request is not None and await request.is_disconnected():
                raise RequestCancelled()
            await asyncio.wait({worker_task}, timeout=0.25)
        return worker_task.result()
    finally:
        if worker_task is None:
            queue.cancel(job)
            queue.finish(job)
        elif not worker_task.done():
            # The thread releases admission only after its in-flight call returns.
            queue.cancel(job)


def _queue_event(job):
    state = REQUEST_QUEUE.snapshot(job)
    if state['state'] == 'queued':
        message = f"正在排队：前方 {state['ahead_count']} 个请求，已等待 {state['queue_wait_ms'] // 1000} 秒"
    else:
        message = '问答请求已开始处理'
    return {'stage': 'request_queue', 'state': 'running' if state['state'] == 'queued' else 'completed',
            'message': message, 'elapsed_ms': state['total_ms'], 'details': state}


@app.post('/api/chat/{request_id}/cancel')
@app.post('/expert-qa/api/chat/{request_id}/cancel')
async def cancel_chat(request_id: str, x_cancel_token: str = Header(default='')):
    try:
        job = REQUEST_QUEUE.cancel_authorized(request_id, x_cancel_token)
    except AdmissionError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    return {'request_id': request_id, **REQUEST_QUEUE.snapshot(job)}


@app.post('/api/chat', response_model=ChatResponse)
@app.post('/expert-qa/api/chat', response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request = None):
    request_id = f'req_{uuid.uuid4().hex}'
    conversation_id = payload.conversation_id or f'conv_{uuid.uuid4().hex}'
    started_at = time.perf_counter()
    job = _admit(payload, request_id, conversation_id)
    try:
        model_name = await run_in_threadpool(partial(_resolve_requested_model, payload.model))
        result = await _execute_job(job, _build_qa_call(payload, model_name=model_name), request, conversation_id=conversation_id)
    except AdmissionError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
    except RequestCancelled as exc:
        raise HTTPException(499, '本次任务已取消。') from exc
    except requests.Timeout as exc:
        raise HTTPException(504, 'The local model timed out while generating a response.') from exc
    except requests.RequestException as exc:
        raise HTTPException(503, 'The local model service is unavailable.') from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        LOGGER.exception('Unhandled chat error for %s', request_id)
        raise HTTPException(500, f'Internal Q&A error ({request_id}).') from exc
    finally:
        if job.state in ('queued', 'running', 'cancelling'):
            REQUEST_QUEUE.cancel(job)
    public = _public_result(result=result, request_id=request_id, conversation_id=conversation_id,
                            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
                            include_debug=payload.include_debug)
    public['retrieval']['request_timing'] = REQUEST_QUEUE.snapshot(job)
    return public


@app.post('/api/chat/stream')
@app.post('/expert-qa/api/chat/stream')
async def chat_stream(payload: ChatRequest):
    """FIFO queue updates and real pipeline progress, with disconnect cancellation."""
    request_id = f'req_{uuid.uuid4().hex}'
    conversation_id = payload.conversation_id or f'conv_{uuid.uuid4().hex}'
    started_at = time.perf_counter()
    job = _admit(payload, request_id, conversation_id)
    try:
        model_name = await run_in_threadpool(partial(_resolve_requested_model, payload.model))
    except requests.RequestException as exc:
        REQUEST_QUEUE.cancel(job)
        raise HTTPException(503, 'The local model service is unavailable.') from exc
    except ValueError as exc:
        REQUEST_QUEUE.cancel(job)
        raise HTTPException(422, str(exc)) from exc
    except BaseException:
        REQUEST_QUEUE.cancel(job)
        raise
    event_queue = asyncio.Queue()
    event_loop = asyncio.get_running_loop()

    def progress_callback(event):
        if not job.cancelled.is_set() and not event_loop.is_closed():
            event_loop.call_soon_threadsafe(event_queue.put_nowait, event)

    qa_call = _build_qa_call(payload, progress_callback=progress_callback, model_name=model_name)

    async def event_stream():
        task = None
        last_queue_update = 0
        last_sent = time.monotonic()
        try:
            yield _sse_event('meta', {'request_id': request_id, 'conversation_id': conversation_id,
                'mode': 'kg-rag', 'model': model_name, 'cancel_token': job.cancel_token,
                'queue': REQUEST_QUEUE.snapshot(job), 'message': _queue_event(job)['message']})
            task = _retain_task(asyncio.create_task(_execute_job(job, qa_call, conversation_id=conversation_id)))
            queue_completed = False
            while True:
                now = time.monotonic()
                state = REQUEST_QUEUE.snapshot(job)['state']
                if state == 'queued' and now - last_queue_update >= 1:
                    yield _sse_event('progress', _queue_event(job))
                    last_queue_update = now
                    last_sent = now
                elif state in ('running', 'cancelling', 'completed', 'failed') and not queue_completed:
                    yield _sse_event('progress', _queue_event(job))
                    queue_completed = True
                    last_sent = now
                if task.done() and event_queue.empty():
                    try:
                        result = task.result()
                    except RequestCancelled:
                        yield _sse_event('error', {'request_id': request_id, 'status_code': 499, 'detail': '本次任务已取消。'})
                    except AdmissionError as exc:
                        yield _sse_event('error', {'request_id': request_id, 'status_code': exc.status_code, 'detail': exc.detail})
                    except Exception as exc:
                        LOGGER.exception('Streaming chat failed for %s', request_id)
                        yield _sse_event('error', _stream_error(exc, request_id))
                    else:
                        public = _public_result(result=result, request_id=request_id, conversation_id=conversation_id,
                            elapsed_ms=round((time.perf_counter() - started_at) * 1000), include_debug=payload.include_debug)
                        public['retrieval']['request_timing'] = REQUEST_QUEUE.snapshot(job)
                        yield _sse_event('result', public)
                    break
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_sent >= STREAM_HEARTBEAT_SECONDS:
                        last_sent = time.monotonic()
                        yield ': keep-alive\n\n'
                    continue
                last_sent = time.monotonic()
                yield _sse_event('progress', event)
        finally:
            if job.state in ('queued', 'running', 'cancelling'):
                REQUEST_QUEUE.cancel(job)
            if task is not None and not task.done():
                task.cancel()  # Cancels scheduling, not the protected worker thread.

    return StreamingResponse(event_stream(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no', 'Connection': 'keep-alive'})
