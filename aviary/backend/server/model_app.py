import asyncio
import traceback
from typing import Any, AsyncGenerator, Dict, Type, Union

from aviary.backend.server.utils import EOS_SENTINELS

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from fastapi import Body, FastAPI
from pydantic import ValidationError as PydanticValidationError
from ray.exceptions import RayActorError
from starlette.requests import Request
from starlette.responses import StreamingResponse

from aviary.backend.llm.engine import TextGenerationInferenceEngine
from aviary.backend.logger import get_logger
from aviary.backend.server.models import (
    Args,
    AviaryModelResponse,
    Prompt,
)
from aviary.backend.server.utils import QueuePriority
from aviary.common.models import (
    ErrorResponse,
)

logger = get_logger(__name__)

model_app = FastAPI()


class LLMDeployment:
    _default_engine_cls: Type[
        TextGenerationInferenceEngine
    ] = TextGenerationInferenceEngine

    def __init__(self, engine=None) -> None:
        self.args = None
        # Keep track of requests to cancel them gracefully
        self.requests_ids: Dict[int, bool] = {}
        self.curr_request_id: int = 0
        self.engine = engine or self._default_engine_cls(None)

    async def reconfigure(
        self,
        config: Union[Dict[str, Any], Args],
    ) -> None:
        logger.info("Reconfiguring...")
        if not isinstance(config, Args):
            new_args: Args = Args.parse_obj(config)
        else:
            new_args: Args = config

        should_reinit_worker_group = True

        self.args = new_args
        self.engine.engine_config = new_args.engine_config
        if should_reinit_worker_group:
            await self.engine.rollover(
                self.args.air_scaling_config,
                pg_timeout_s=self.args.scaling_config.pg_timeout_s,
            )
        logger.info("Reconfigured and ready to serve.")

    # TODO Remove once we can stream from serve handles
    @model_app.get("/metadata")
    async def metadata(self) -> dict:
        return {
            "metadata": self.args.dict(
                include={
                    "engine_config": {
                        "generation",
                        "model_id",
                        "model_url",
                        "model_description",
                    }
                }
            )
        }

    # TODO Remove once we can stream from serve handles
    @model_app.post("/stream")
    async def generate_text_stream(
        self, prompt: Prompt, request: Request, priority: Annotated[int, Body()]
    ) -> StreamingResponse:
        self.engine.validate_prompt(prompt)

        async def wrapper():
            """Wrapper to always yield json-formatted strings"""
            try:
                async for t in self.generate_tokens(
                    prompt,
                    request,
                    priority=QueuePriority(priority),
                ):
                    if t in EOS_SENTINELS:
                        continue
                    yield t.json() + "\n"
            except Exception as e:
                message = "".join(traceback.format_exception_only(type(e), e)).strip()
                # TODO clean this up
                if isinstance(e, PydanticValidationError):
                    error_code = 400
                else:
                    error_code = getattr(e, "error_code", None)
                if error_code:
                    logger.warning(
                        f"Validation error caught while generating: {message}"
                    )
                else:
                    logger.error(
                        f"Exception caught while generating:\n{traceback.format_exc()}"
                    )
                yield AviaryModelResponse(
                    error=ErrorResponse(
                        message=message,
                        internal_message=message,
                        code=error_code or 500,
                        type=e.__class__.__name__,
                    ),
                ).json() + "\n"

        return StreamingResponse(wrapper(), media_type="text/event-stream")

    async def _generate_token_stream(
        self,
        prompt: Prompt,
        request_id: int,
        priority: QueuePriority,
    ) -> AsyncGenerator[Union[AviaryModelResponse, Type[StopIteration]], None]:
        while not self.engine.is_initialized():
            logger.info("Waiting for worker group to be initialized...")
            await asyncio.sleep(1)

        try:
            async for result in self.engine.stream_async(prompt, priority):
                yield result if result is not None or self.requests_ids[
                    request_id
                ] else StopIteration
        except RayActorError as e:
            raise RuntimeError(
                f"Prediction failed due to RayActorError. "
                "This usually means that one or all prediction workers are dead. "
                "Try again in a few minutes. "
                f"Traceback:\n{traceback.print_exc()}"
            ) from e

    # Called by Serve to check the replica's health.
    async def check_health(self):
        self.engine.check_health()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.args.engine_config.model_id}"

    async def generate_tokens(
        self,
        prompt: Prompt,
        request: Request,
        priority: QueuePriority,
    ) -> AsyncGenerator[Union[AviaryModelResponse, Type[StopIteration]], None]:
        """Generate text from the given prompts in batch.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            request (Request): Request object.
            **kwargs: Additional arguments to pass to the batch function.
        """
        curr_request_id = self.curr_request_id
        self.requests_ids[curr_request_id] = False
        self.curr_request_id += 1
        async_generator = self._generate_token_stream(
            prompt,
            curr_request_id,
            priority,
        )
        try:
            async for result in async_generator:
                yield result
                if await request.is_disconnected():
                    self.requests_ids[curr_request_id] = True
                    await async_generator.aclose()
                    break
        finally:
            del self.requests_ids[curr_request_id]
