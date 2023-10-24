from contextlib import contextmanager
from typing import Generator, Optional

from opentelemetry import context
from opentelemetry.context.context import Context

# Sentinel value representing "use the current context"
CURRENT_CONTEXT = Context()


@contextmanager
def use_context(
    parent_context: Optional[Context] = CURRENT_CONTEXT,
) -> Generator[None, None, None]:
    if parent_context is CURRENT_CONTEXT:
        # Do nothing, default behavior of opentelemetry is to use the current context
        yield
    else:
        new_context = parent_context if parent_context is not None else Context()
        token = context.attach(new_context)
        try:
            yield
        finally:
            context.detach(token)
