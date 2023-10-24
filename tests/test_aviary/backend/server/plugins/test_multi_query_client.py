import asyncio
from unittest.mock import Mock

import pytest
from asynctest import CoroutineMock
from fastapi import HTTPException
from starlette.requests import Request

from rayllm.backend.server.models import AviaryModelResponse, Prompt, QueuePriority
from rayllm.backend.server.plugins.execution_hooks import ExecutionHooks
from rayllm.backend.server.plugins.multi_query_client import MultiQueryClient
from rayllm.backend.server.plugins.router_query_engine import RouterQueryClient


class FakeRouterQueryClient(RouterQueryClient):
    def __init__(self, *models: str):
        self._models = models

    async def stream(
        self, model: str, prompt: Prompt, request: Request, priority: QueuePriority
    ):
        for m in self._models:
            yield AviaryModelResponse(generated_text=m, num_generated_tokens=1)
            await asyncio.sleep(0.01)

    async def model(self, model: str):
        if model in self._models:
            return model
        return None

    async def models(self):
        return {v: v for v in self._models}


class FakeHook:
    def __init__(self):
        self.count = 0
        self._args = None

    def trigger(self, *args, **kwargs):
        self.count += 1
        self._args = (args, kwargs)


@pytest.mark.asyncio
async def test_multi_query_client():
    models = [f"model_{i}" for i in range(10)]
    clients = [FakeRouterQueryClient(m) for m in models]

    mock_hook = Mock()
    hooks = ExecutionHooks()
    hooks.add_post_execution_hook(mock_hook)

    multi_client = MultiQueryClient(*clients, hooks=hooks)
    mc_models = await multi_client.models()
    assert mc_models == {m: m for m in models}

    for model in models:
        assert await multi_client.model(model) == model

    assert await multi_client.model("model_11") is None


@pytest.mark.asyncio
async def test_multi_query_client_stream():
    models = [f"model_{i}" for i in range(10)]
    clients = [FakeRouterQueryClient(m) for m in models]

    mock_hook = CoroutineMock()
    hooks = ExecutionHooks()
    hooks.add_post_execution_hook(mock_hook)

    multi_client = MultiQueryClient(*clients, hooks=hooks)
    models = [f"model_{i}" for i in range(10)]

    miter = iter(models)
    for model in models:
        request = Mock()
        async for x in multi_client.stream(
            model, Prompt(prompt="prompt"), request, QueuePriority.GENERATE_TEXT
        ):
            # There should only be one response
            assert x.generated_text == next(miter)

        # Wait for the execution hook to run
        await asyncio.sleep(0.01)
        mock_hook.assert_called_once_with(request, model, "prompt", x, True)
        mock_hook.reset_mock()

    with pytest.raises(HTTPException):
        async for _ in multi_client.stream(
            "fake", Prompt(prompt="prompt"), request, QueuePriority.GENERATE_TEXT
        ):
            pytest.fail("Should not have streamed any tokens")
