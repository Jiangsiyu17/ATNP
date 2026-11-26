"""Request-scoped wall-clock telemetry. Never records prompts, answers or evidence.

Spans are inclusive and may nest; sum only model-call durations for LLM wall time.
HTTP duration includes server scheduling, not just GPU computation. No deadlines.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import threading
import time
import uuid

_identity = ContextVar('performance_identity', default=None)
_trace = ContextVar('performance_trace', default=None)
_parent = ContextVar('performance_parent', default=None)
_stage = ContextVar('performance_stage', default='unclassified')
_sink_lock = threading.Lock()
_sinks = {}
LOGGER = logging.getLogger(__name__)


@contextmanager
def request_identity(request_id, conversation_id=None):
    token = _identity.set((request_id, conversation_id))
    try:
        yield
    finally:
        _identity.reset(token)


class Trace:
    def __init__(self):
        self.request_id, self.conversation_id = _identity.get() or ('perf_' + uuid.uuid4().hex, None)
        self.started = time.perf_counter()
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.spans, self.calls, self.events = [], [], []
        self.lock = threading.Lock()
        self.sequence = 0
        self.model_sequence = 0

    def next_id(self):
        with self.lock:
            self.sequence += 1
            return self.sequence

    def snapshot(self, status, result=None, error_type=None):
        retrieval = (result or {}).get('retrieval') or {}
        validations = [e for e in self.events if e['stage'] == 'claim_validation' and e['state'] == 'completed']
        return {
            'schema_version': 1, 'request_id': self.request_id,
            'conversation_id': self.conversation_id, 'created_at': self.created_at,
            'status': status, 'error_type': error_type,
            'pipeline_wall_ms': round((time.perf_counter() - self.started) * 1000, 3),
            'llm_call_count': len(self.calls),
            'llm_http_wall_ms': round(sum(c['duration_ms'] for c in self.calls), 3),
            'usage_missing_call_count': sum(c.get('prompt_tokens') is None or c.get('completion_tokens') is None for c in self.calls),
            'reported_prompt_tokens': sum(c.get('prompt_tokens') or 0 for c in self.calls),
            'reported_completion_tokens': sum(c.get('completion_tokens') or 0 for c in self.calls),
            'generation_attempts': retrieval.get('generation_attempts'),
            'answer_repaired': retrieval.get('answer_repaired'),
            'safe_fallback_applied': retrieval.get('safe_fallback_applied'),
            'first_claim_validation_valid': validations[0].get('valid') if validations else None,
            'claim_validation_rounds': len(validations),
            'spans': self.spans, 'model_calls': self.calls, 'events': self.events,
        }


@contextmanager
def span(stage, *, kind='stage', **metadata):
    trace = _trace.get()
    if trace is None:
        yield {}
        return
    started = time.perf_counter()
    row = {'span_id': trace.next_id(), 'parent_span_id': _parent.get(),
           'stage': stage, 'kind': kind, 'start_ms': round((started - trace.started) * 1000, 3),
           'status': 'ok', **metadata}
    if kind == 'llm':
        with trace.lock:
            trace.model_sequence += 1
            row['call_index'] = trace.model_sequence
    parent_token, stage_token = _parent.set(row['span_id']), _stage.set(stage)
    try:
        yield row
    except BaseException as exc:
        row.update(status='cancelled' if type(exc).__name__ == 'RequestCancelled' else 'error',
                   error_type=type(exc).__name__)
        raise
    finally:
        row['duration_ms'] = round((time.perf_counter() - started) * 1000, 3)
        _parent.reset(parent_token)
        _stage.reset(stage_token)
        with trace.lock:
            (trace.calls if kind == 'llm' else trace.spans).append(row)


def model_span(**metadata):
    return span(_stage.get(), kind='llm', **metadata)


def timed_call(stage, function, *args, **kwargs):
    with span(stage):
        return function(*args, **kwargs)


def timed(stage):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return timed_call(stage, function, *args, **kwargs)
        return wrapped
    return decorate


def record_progress(stage, state, details):
    trace = _trace.get()
    if trace is None:
        return
    # No message text, question, evidence or free-form reason enters the sink.
    fields = ('valid', 'total_claims', 'supported_claims', 'tool_call_count',
              'recovery_count', 'reference_count', 'evidence_count')
    row = {'stage': stage, 'state': state, 'at_ms': round((time.perf_counter()-trace.started)*1000, 3)}
    row.update({k: details[k] for k in fields if isinstance(details.get(k), (int, bool))})
    with trace.lock:
        trace.events.append(row)


def _persist(report):
    destination = os.getenv('ATNP_PERFORMANCE_LOG_PATH', '').strip()
    if not destination:
        return
    try:
        with _sink_lock:
            handler = _sinks.get(destination)
            if handler is None:
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                handler = RotatingFileHandler(destination, maxBytes=10*1024*1024,
                                              backupCount=3, encoding='utf-8')
                handler.setFormatter(logging.Formatter('%(message)s'))
                _sinks[destination] = handler
            record = logging.LogRecord('atnp.performance', logging.INFO, '', 0,
                                       json.dumps(report, ensure_ascii=False), (), None)
            handler.handle(record)
    except Exception:
        # Disk full or unavailable telemetry must not discard a valid answer.
        LOGGER.warning('Could not persist performance telemetry', exc_info=True)


def profile_qa(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        trace = Trace()
        token, parent_token, stage_token = _trace.set(trace), _parent.set(None), _stage.set('qa')
        result, error_type, status = None, None, 'error'
        try:
            result = function(*args, **kwargs)
            status = result.get('status', 'unknown')
            return result
        except BaseException as exc:
            error_type = type(exc).__name__
            status = 'cancelled' if error_type == 'RequestCancelled' else 'error'
            raise
        finally:
            report = trace.snapshot(status, result, error_type)
            if result is not None:
                result.setdefault('retrieval', {})['performance'] = report
            _trace.reset(token)
            _parent.reset(parent_token)
            _stage.reset(stage_token)
            _persist(report)
    return wrapped
