import asyncio
from typing import AsyncIterator, Dict, List, Optional, Tuple

from fastapi import HTTPException, status
from starlette.requests import Request

from aviary.backend.logger import get_logger
from aviary.backend.observability.base import step
from aviary.backend.server.models import (
    AviaryModelResponse,
    Prompt,
    QueuePriority,
)
from aviary.backend.server.plugins.execution_hooks import (
    ExecutionHooks,
)
from aviary.backend.server.plugins.router_query_engine import (
    RouterQueryClient,
    StreamingErrorHandler,
)
from aviary.common.models import ModelData

logger = get_logger(__name__)


class MultiQueryClient(RouterQueryClient):
    """A RouterQueryClient that combines other RouterQueryClients

    This client iterates over the other clients it includes and forwards requests to them.
    """

    def __init__(
        self,
        *clients: RouterQueryClient,
        hooks: Optional[ExecutionHooks] = None,
    ) -> None:
        self.clients = clients
        self.metrics_wrapper = StreamingErrorHandler(hooks)

    async def stream(
        self,
        model: str,
        prompt: Prompt,
        request: Request,
        priority: QueuePriority,
    ) -> AsyncIterator[AviaryModelResponse]:
        client, _ = await self._find_client_for_model(model)
        if not client:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"Unable to find {model}. Please ensure that the model exists and you have permission.",
            )

        with step(
            "aviary_request",
            request.state.request_id,
            baggage={"model_id": model},
        ) as span:
            request.state.aviary_request_span = span

            async for x in self.metrics_wrapper.handle_failure(
                model,
                request,
                prompt,
                client.stream(model, prompt, request, priority),
            ):
                yield x

    async def _find_client_for_model(
        self, model: str, raise_if_missing=True
    ) -> Tuple[RouterQueryClient, ModelData]:
        for client in self.clients:
            model_def = await client.model(model)
            if model_def:
                return client, model_def
        return None, None

    async def models(self):
        all_model_data: List[Dict[str, ModelData]] = await asyncio.gather(
            *(client.models() for client in self.clients)
        )

        # If a model is supported twice, prioritize the data from the earlier client
        return {
            k: v
            for model_data in reversed(all_model_data)
            for k, v in model_data.items()
        }

    async def model(self, model_id: str):
        _, model_data = await self._find_client_for_model(model_id)
        return model_data
