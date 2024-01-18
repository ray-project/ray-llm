import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional

from starlette.requests import Request

from rayllm.backend.logger import get_logger
from rayllm.backend.observability.fn_call_metrics import (
    InstrumentTokenAsyncGenerator,
)
from rayllm.backend.server.metrics import Metrics
from rayllm.backend.server.models import (
    AviaryModelResponse,
    Prompt,
    QueuePriority,
)
from rayllm.backend.server.plugins.execution_hooks import (
    ExecutionHooks,
    ShieldedTaskSet,
)
from rayllm.backend.server.utils import get_response_for_error
from rayllm.common.models import DeletedModel, ModelData

logger = get_logger(__name__)


class RouterQueryClient(ABC):
    @abstractmethod
    async def stream(
        self, model: str, prompt: Prompt, request: Request, priority: QueuePriority
    ) -> AsyncIterator[AviaryModelResponse]:
        """Stream a response from a specific model"""
        ...

    async def query(self, model: str, prompt: Prompt, request: Request):
        response_stream = self.stream(
            model,
            prompt,
            request,
            priority=QueuePriority.BATCH_GENERATE_TEXT,
        )
        responses = [resp async for resp in response_stream]
        return AviaryModelResponse.merge_stream(*responses)

    @abstractmethod
    async def models(self) -> Dict[str, ModelData]:
        """Get configurations for supported models"""
        ...

    @abstractmethod
    async def model(self, model_id: str) -> Optional[ModelData]:
        """Get configurations for a supported model"""
        ...

    @abstractmethod
    async def delete_fine_tuned_model(self, model: str) -> DeletedModel:
        """Delete a fine-tuned model"""
        ...


class StreamingErrorHandler:
    """Handle errors and finalizers for an AviaryModelResponse stream.

    This class:
    1. Tracks request level metrics for the response stream
    2. Handles errors in the router level code for the response stream
    3. Registers finalizers [eg. execute post execution hooks]
    """

    def __init__(
        self,
        hooks: Optional[ExecutionHooks] = None,
        metrics: Optional[Metrics] = None,
        task_set: Optional[ShieldedTaskSet] = None,
    ):
        self.hooks = hooks or ExecutionHooks()
        self.metrics = metrics or Metrics()
        self.task_set = task_set or ShieldedTaskSet()

    @InstrumentTokenAsyncGenerator("router_get_response_stream")
    async def handle_failure(
        self,
        model: str,
        request: Request,
        prompt: Prompt,
        async_iterator: AsyncIterator[AviaryModelResponse],
    ):
        req_id = request.state.request_id
        model_tags = {"model_id": model}
        self.metrics.requests_started.inc(tags=model_tags)
        is_first_token = True

        responses: List[AviaryModelResponse] = []
        try:
            async with self.hooks.context():
                async for response in async_iterator:
                    responses.append(response)

                    # Track metrics per token
                    self.metrics.track(response, is_first_token, model)
                    is_first_token = False
                    yield response
        except asyncio.CancelledError:
            # NOTE: We just log cancellation and re-throw it immediately to interrupt
            #       request handling
            logger.warning(f"Request ({req_id}) has been cancelled")
            raise
        except Exception as e:
            logger.error(
                f"Failed while streaming back a response for request ({req_id}): {repr(e)}",
                exc_info=e,
            )

            yield get_response_for_error(e, request_id=req_id)
            # DO NOT RAISE.
            # We do not raise here because that would cause a disconnection for streaming.
        finally:
            # Merge the responses
            if responses:
                merged_response = AviaryModelResponse.merge_stream(*responses)
                logger.info(f"Recording to post execution hook. Id: {req_id}")
                # Trigger the post execution hooks once
                # Shield from asyncio cancellation
                await self.task_set.run(
                    self.hooks.trigger_post_execution_hook(
                        request, model, prompt.prompt, True, merged_response
                    )
                )
            else:
                logger.info(f"No tokens produced. Id: {req_id}")
