import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, List, Optional

from fastapi import HTTPException, status
from opentelemetry import trace
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request

from rayllm.backend.llm.generation import FinishReason
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
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.plugins.execution_hooks import (
    ExecutionHooks,
    ShieldedTaskSet,
)
from rayllm.backend.server.utils import extract_message_from_exception
from rayllm.common.models import ErrorResponse, ModelData

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


class StreamingErrorHandler:
    """Handle errors and finalizers for an AviaryModelResopnse stream.

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
        except asyncio.CancelledError as e:
            # The request is cancelled. Try to return a last Aviary Model response, then raise
            # We raise here because we don't want to interrupt the cancellation
            yield _get_response_for_error(e, request_id=req_id)
            raise
        except Exception as e:
            # Something went wrong.
            yield _get_response_for_error(e, request_id=req_id)
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


def _get_response_for_error(e: Exception, request_id: str):
    """Convert an exception to an AviaryModelResponse object"""
    logger.error(f"Request {request_id} failed with:", exc_info=e)
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(e, HTTPException):
        status_code = e.status_code
    elif isinstance(e, OpenAIHTTPException):
        status_code = e.status_code
    elif isinstance(e, PydanticValidationError):
        status_code = 400
    else:
        # Try to get the status code attribute
        status_code = getattr(e, "status_code", status_code)

    if status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
        message = "Internal Server Error"
        exc_type = "InternalServerError"
    else:
        message = extract_message_from_exception(e)
        exc_type = e.__class__.__name__

    message += f" (Request ID: {request_id})"

    span = trace.get_current_span()
    span.record_exception(e)
    span.set_status(trace.StatusCode.ERROR, description=message)

    return AviaryModelResponse(
        error=ErrorResponse(
            message=message,
            code=status_code,
            internal_message=message,
            type=exc_type,
        ),
        finish_reason=FinishReason.ERROR,
    )
