"""Process-local FIFO admission; use one API worker with the shared model server."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import secrets
import math
import threading
import time


class RequestCancelled(BaseException):
    """Control flow: must escape broad Exception handlers in retrieval/repair code."""


_cancellation = ContextVar('atnp_request_cancellation', default=None)


def check_cancelled():
    event = _cancellation.get()
    if event is not None and event.is_set():
        raise RequestCancelled()


@contextmanager
def cancellation_scope(event):
    token = _cancellation.set(event)
    try:
        check_cancelled()
        yield
        check_cancelled()
    finally:
        _cancellation.reset(token)


class AdmissionError(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code, self.detail = status_code, detail


@dataclass
class Ticket:
    request_id: str
    key: str
    created: float
    cancel_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    cancelled: threading.Event = field(default_factory=threading.Event)
    state: str = 'queued'
    started: float | None = None
    ended: float | None = None


class RequestQueue:
    def __init__(self, max_waiting=4, wait_timeout=0, clock=time.monotonic, max_running=1):
        if max_waiting < 0 or not math.isfinite(wait_timeout) or wait_timeout < 0:
            raise ValueError('Queue capacity and timeout must be nonnegative; zero disables queue expiry')
        if type(max_running) is not int or max_running < 1:
            raise ValueError('Execution capacity must be a positive integer')
        self.max_waiting, self.wait_timeout, self.clock = max_waiting, wait_timeout, clock
        self.max_running = max_running
        self.lock = threading.Lock()
        self.waiting, self.jobs, self.keys = [], {}, {}
        self.running = {}

    @property
    def active(self):
        """Compatibility view for the original single-worker callers."""
        return next(iter(self.running.values()), None)

    def _remove(self, job, state):
        job.state, job.ended = state, self.clock()
        if job in self.waiting:
            self.waiting.remove(job)
        self.jobs.pop(job.request_id, None)
        self.keys.pop(job.key, None)

    def _expire(self):
        if self.wait_timeout == 0:
            return  # Bounded capacity + explicit cancellation, without premature expiry.
        now = self.clock()
        for job in list(self.waiting):
            if now - job.created >= self.wait_timeout:
                self._remove(job, 'queue_timeout')

    def submit(self, request_id, key):
        with self.lock:
            self._expire()
            if key in self.keys:
                raise AdmissionError(409, '相同问题已在当前对话中排队或处理，请勿重复提交。')
            # Includes execution slots, even before their coroutines start.
            if len(self.jobs) >= self.max_waiting + self.max_running:
                raise AdmissionError(429, '当前问答队列已满，请稍后重试。')
            job = Ticket(request_id, key, self.clock())
            self.waiting.append(job)
            self.jobs[request_id] = job
            self.keys[key] = request_id
            return job

    def try_start(self, job):
        with self.lock:
            self._expire()
            if job.state == 'queued' and len(self.running) < self.max_running and self.waiting[0] is job:
                self.waiting.pop(0)
                self.running[job.request_id] = job
                job.state, job.started = 'running', self.clock()
                return True
            return False

    def cancel(self, job):
        with self.lock:
            job.cancelled.set()
            if job.state == 'queued':
                self._remove(job, 'cancelled')
            elif self.running.get(job.request_id) is job:
                job.state = 'cancelling'  # Do NOT release a still-running GPU call.

    def cancel_authorized(self, request_id, token):
        with self.lock:
            job = self.jobs.get(request_id)
            if job is None or not token.isascii() or not secrets.compare_digest(job.cancel_token, token):
                raise AdmissionError(404, '任务不存在或取消凭据无效。')
        self.cancel(job)
        return job

    def finish(self, job, failed=False):
        with self.lock:
            if self.running.get(job.request_id) is job:
                self.running.pop(job.request_id)
            if job.request_id in self.jobs:
                self._remove(job, 'cancelled' if job.cancelled.is_set() else 'failed' if failed else 'completed')

    def snapshot(self, job):
        with self.lock:
            self._expire()
            now = job.ended if job.ended is not None else self.clock()
            ahead = (self.waiting.index(job) + len(self.running)) if job in self.waiting else 0
            return {'state': job.state, 'ahead_count': ahead,
                    'waiting_count': len(self.waiting), 'max_waiting': self.max_waiting,
                    'running_count': len(self.running), 'max_running': self.max_running,
                    'queue_timeout_seconds': self.wait_timeout,
                    'queue_wait_ms': round(((job.started if job.started is not None else now) - job.created) * 1000),
                    'processing_ms': round((now - job.started) * 1000) if job.started is not None else 0,
                    'total_ms': round((now - job.created) * 1000)}
