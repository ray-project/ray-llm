import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from functools import wraps
from typing import Coroutine

from opentelemetry import trace
from starlette.requests import Request

from rayllm.backend.server.models import (
    AviaryModelResponse,
)

tracer = trace.get_tracer(__name__)


class ExecutionHooks:
    def __init__(self, hooks=None, context_managers=None):
        self.hooks = hooks or []
        self.context_managers = context_managers or []

    def add_post_execution_hook(self, fn):
        self.hooks.append(fn)

    def add_context_manager(self, fn):
        self.context_managers.append(fn)

    @asynccontextmanager
    async def context(self):
        # Enter the provided context managers
        async with AsyncExitStack() as stack:
            for cm in self.context_managers:
                await stack.enter_async_context(cm())
            yield

    async def trigger_post_execution_hook(
        self,
        request: Request,
        model_id: str,
        input_str: str,
        is_first_token: bool,
        output: AviaryModelResponse,
    ):
        # Run the token hooks in parallel
        if len(self.hooks) > 0:
            with tracer.start_as_current_span("trigger_post_execution_hook"):
                await asyncio.gather(
                    *[
                        self.wrap_with_trace(fn)(
                            request,
                            model_id,
                            input_str,
                            output,
                            is_first_token,
                        )
                        for fn in self.hooks
                    ]
                )

    @staticmethod
    def wrap_with_trace(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(fn.__name__):
                return await fn(*args, **kwargs)

        return wrapper


class ShieldedTaskSet:
    """A task set that runs a coroutine with shielding.

    This task set will ensure that a coroutine runs by wrapping it
    in a task, and shielding it from parent cancellations.

    Note: this class does not protect against event loop failure.
    """

    def __init__(self):
        self.task_set = set()

    def run(self, coroutine: Coroutine) -> asyncio.Future:
        # Wrap the coroutine in a task so it executes immediately
        # Save a reference to the task so it doesn't get garbage collected
        # Remove the task from the task set when it is completed
        task = asyncio.create_task(coroutine)
        self.task_set.add(task)
        task.add_done_callback(self.task_set.discard)

        # Shield the task so it cannot be cancelled by the parent
        shielded_task = asyncio.shield(task)

        # Return the shielded task so it can be awaited by the parent.
        return shielded_task
