from fastapi import Request
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


async def add_request_id(request: Request, call_next):
    request.state.request_id = trace.format_trace_id(
        trace.get_current_span().get_span_context().trace_id
    )

    return await call_next(request)
