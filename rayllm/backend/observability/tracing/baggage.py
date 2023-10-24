from contextlib import contextmanager
from typing import Generator, Mapping

from opentelemetry.baggage import get_baggage, set_baggage
from opentelemetry.context import get_current
from opentelemetry.util.types import AttributeValue

from rayllm.backend.observability.tracing.context import use_context


@contextmanager
def baggage(attributes: Mapping[str, AttributeValue]) -> Generator[None, None, None]:
    """
    A context manager that sets multiple baggage values on the active context.
    This can be combined with the BaggageSpanProcessor to add the baggage values
    to the span as attributes.

    Example:

    with baggage({"foo": "bar"}):
        do_something()

    """
    ctx = get_current()
    for key, value in attributes.items():
        ctx = set_baggage(key, value, context=ctx)

    with use_context(ctx):
        yield


def get(key: str):
    return get_baggage(key)
