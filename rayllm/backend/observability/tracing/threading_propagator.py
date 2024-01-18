import concurrent.futures.thread
from functools import wraps
from typing import Collection

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import (
    get_current_span,
)

from rayllm.backend.observability.tracing.context import use_context


class _InstrumentedThreadPoolExecutorWorkItem(concurrent.futures.thread._WorkItem):
    def __init__(self, future, fn, args, kwargs):
        super().__init__(
            future,
            self.wrap_with_context(fn, parent_span=get_current_span()),
            args,
            kwargs,
        )

    @staticmethod
    def wrap_with_context(fn, parent_span: trace.Span):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with use_context(trace.set_span_in_context(parent_span)):
                return fn(*args, **kwargs)

        return wrapper


class ThreadPoolExecutorInstrumentor(BaseInstrumentor):
    original_workitem = concurrent.futures.thread._WorkItem

    def instrumentation_dependencies(self) -> Collection[str]:
        return ()

    def _instrument(self, *args, **kwargs):
        concurrent.futures.thread._WorkItem = _InstrumentedThreadPoolExecutorWorkItem

    def _uninstrument(self, **kwargs):
        concurrent.futures.thread._WorkItem = self.original_workitem
