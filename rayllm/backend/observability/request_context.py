import contextvars
import weakref
from contextlib import contextmanager

from fastapi.datastructures import State

from rayllm.backend.observability.tracing import baggage

# Fast api state context.
# This may include secrets
_fastapi_state_context: contextvars.ContextVar[
    weakref.ReferenceType[State]
] = contextvars.ContextVar("aviary_fastapi_state")


def set(**kwargs):
    return baggage.baggage(kwargs)


def get(key: str):
    return baggage.get(key)


@contextmanager
def set_fastapi_state(request_state: State):
    # Hold a weakref to make sure that we clean up any state once Fastapi request object is garbage collected.
    ref = weakref.ref(request_state)
    token = _fastapi_state_context.set(ref)
    try:
        yield
    finally:
        _fastapi_state_context.reset(token)


def get_fastapi_state():
    ctx = _fastapi_state_context.get(None)
    if ctx:
        return ctx()


def maybe_get_string_field(field: str):
    state = get_fastapi_state()
    val = getattr(state, field, None)
    return val if isinstance(val, str) else None
