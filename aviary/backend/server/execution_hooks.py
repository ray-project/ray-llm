import asyncio

from starlette.requests import Request

from aviary.backend.server.models import (
    AviaryModelResponse,
)


class ExecutionHooks:
    def __init__(self):
        self.hooks = []

    def add_post_execution_hook(self, fn):
        self.hooks.append(fn)

    async def trigger_post_execution_hook(
        self,
        request: Request,
        model_id: str,
        input_str: str,
        is_first_token: bool,
        output: AviaryModelResponse,
    ):
        # Run the token hooks in parallel
        # If a token hook fails, the request will fail

        if len(self.hooks) > 0:
            await asyncio.gather(
                *[
                    fn(request, model_id, input_str, output, is_first_token)
                    for fn in self.hooks
                ]
            )
