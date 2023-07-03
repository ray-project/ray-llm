import asyncio
import traceback
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import async_timeout
from fastapi import FastAPI
from ray import serve
from ray.exceptions import RayActorError
from starlette.requests import Request
from starlette.responses import StreamingResponse

from aviary.backend.llm.predictor import ContinuousBatchingPredictor, LLMPredictor
from aviary.backend.logger import get_logger
from aviary.backend.server.batch import QueuePriority, _PriorityBatchQueue
from aviary.backend.server.exceptions import PromptTooLongError
from aviary.backend.server.models import (
    Args,
    DeepSpeed,
    ErrorResponse,
    Prompt,
    Response,
)
from aviary.common.constants import GATEWAY_TIMEOUT_S

EOS_SENTINELS = (None, StopIteration, StopAsyncIteration)
logger = get_logger(__name__)

app = FastAPI()


async def _until_disconnected(request: Request):
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(1)


@serve.ingress(app)
class LLMDeployment(ABC):
    _predictor_cls: Type[LLMPredictor] = LLMPredictor

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.args = None
        # Keep track of requests to cancel them gracefully
        self.requests_ids: Dict[int, bool] = {}
        self.curr_request_id: int = 0
        self.predictor = self._predictor_cls(
            self.args.model_config if self.args else None
        )

    @abstractmethod
    def _should_reinit_worker_group(self, new_args: Args) -> bool:
        pass

    async def reconfigure(
        self,
        config: Union[Dict[str, Any], Args],
        force: bool = False,
    ) -> None:
        logger.info("Reconfiguring...")
        if not isinstance(config, Args):
            new_args: Args = Args.parse_obj(config)
        else:
            new_args: Args = config

        should_reinit_worker_group = force or self._should_reinit_worker_group(new_args)

        self.args = new_args
        self.predictor.model_config = new_args.model_config
        if should_reinit_worker_group:
            await self.predictor.rollover(
                self.args.air_scaling_config,
                pg_timeout_s=self.args.scaling_config.pg_timeout_s,
            )
        logger.info("Reconfigured.")

    async def validate_prompt(self, prompt: Prompt) -> None:
        if len(prompt.prompt.split()) > self.args.model_config.max_input_words:
            raise PromptTooLongError(
                f"Prompt exceeds max input words of "
                f"{self.args.model_config.max_input_words}. "
                "Please make the prompt shorter."
            )

    @app.get("/metadata")
    async def metadata(self) -> dict:
        return {
            "metadata": self.args.dict(
                exclude={
                    "model_config": {
                        "initialization": {"s3_mirror_config", "runtime_env"}
                    }
                }
            )
        }

    @app.post("/query")
    async def generate_text(self, prompt: Prompt, request: Request) -> Response:
        return await self._generate_text(
            prompt, request, priority=QueuePriority.GENERATE_TEXT
        )

    async def _generate_text(
        self, prompt: Prompt, request: Request, *, priority: QueuePriority
    ) -> Response:
        await self.validate_prompt(prompt)
        with async_timeout.timeout(GATEWAY_TIMEOUT_S):
            responses = []
            async for t in self.generate_text_batch(
                prompt,
                request,
                priority=priority,
                # start_timestamp=start_timestamp,
            ):
                if isinstance(t, list):
                    t = t[0]
                if t in EOS_SENTINELS:
                    continue
                responses.append(t)
            return Response.merge_stream(*responses)

    @app.post("/stream")
    async def generate_text_stream(
        self, prompt: Prompt, request: Request
    ) -> StreamingResponse:
        await self.validate_prompt(prompt)

        async def wrapper():
            """Wrapper to always yield json-formatted strings"""
            try:
                async for t in self.generate_text_batch(
                    prompt,
                    request,
                    priority=QueuePriority.GENERATE_TEXT,
                    # start_timestamp=start_timestamp,
                ):
                    if isinstance(t, (list, tuple)):
                        if t[0] in EOS_SENTINELS:
                            continue
                        yield t[0].json() + "\n"
                    else:
                        if t in EOS_SENTINELS:
                            continue
                        yield t.json() + "\n"
            except Exception as e:
                yield ErrorResponse(
                    error="".join(traceback.format_exception_only(type(e), e)).strip(),
                    error_type=e.__class__.__name__,
                ).json() + "\n"
                raise

        return StreamingResponse(wrapper(), status_code=200, media_type="text/plain")

    @app.post("/batch")
    async def batch_generate_text(
        self, prompts: List[Prompt], request: Request
    ) -> List[Response]:
        texts = await asyncio.gather(
            *[
                self._generate_text(
                    prompt,
                    request,
                    priority=QueuePriority.BATCH_GENERATE_TEXT,
                    # start_timestamp=start_timestamp,
                )
                for prompt in prompts
            ]
        )
        return texts

    @abstractmethod
    async def generate_text_batch(
        self,
        prompt: Prompt,
        request: Request,
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
        **kwargs,
    ) -> AsyncGenerator:
        pass

    async def _generate_text_stream(
        self,
        prompts_and_request_ids: List[Tuple[Prompt, int]],
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
        **kwargs,
    ):
        prompts, request_ids = zip(*prompts_and_request_ids)
        if isinstance(start_timestamp, list) and start_timestamp[0]:
            start_timestamp = min(start_timestamp)
        elif isinstance(start_timestamp, list):
            start_timestamp = start_timestamp[0]
        if isinstance(timeout_s, list) and timeout_s[0]:
            timeout_s = min(timeout_s)
        elif isinstance(timeout_s, list):
            timeout_s = timeout_s[0]

        logger.info(
            f"Received {len(prompts)} prompts {prompts} request_ids {request_ids}. start_timestamp {start_timestamp} timeout_s {timeout_s}"
        )

        while not self.predictor.is_initialized():
            logger.info("Waiting for worker group to be initialized...")
            await asyncio.sleep(1)

        try:
            logger.info(f"calling _stream_async on {request_ids}")
            async for result in self.predictor._stream_async(
                prompts,
                timeout_s=timeout_s,
                start_timestamp=start_timestamp,
            ):
                yield [
                    v if v is not None or self.requests_ids[id] else StopIteration
                    for v, id in zip(result, request_ids)
                ]
        except RayActorError as e:
            raise RuntimeError(
                f"Prediction failed due to RayActorError. "
                "This usually means that one or all prediction workers are dead. "
                "Try again in a few minutes. "
                f"Traceback:\n{traceback.print_exc()}"
            ) from e
        finally:
            logger.info(f"Batch for {request_ids} finished")

    # Called by Serve to check the replica's health.
    async def check_health(self):
        self.predictor.check_health()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.args.model_config.model_id}"


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    max_concurrent_queries=10,  # Maximum backlog for a single replica
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class StaticBatchingLLMDeployment(LLMDeployment):
    def _should_reinit_worker_group(self, new_args: Args) -> bool:
        old_args = self.args

        if not old_args:
            return True

        old_scaling_config = self.args.air_scaling_config
        new_scaling_config = new_args.scaling_config.as_air_scaling_config()

        if not self.predictor.is_initialized():
            return True

        if old_scaling_config != new_scaling_config:
            return True

        if not old_args:
            return True

        if old_args.model_config.initialization != new_args.model_config.initialization:
            return True

        if (
            old_args.model_config.generation.max_batch_size
            != new_args.model_config.generation.max_batch_size
            and isinstance(new_args.model_config.initialization.initializer, DeepSpeed)
        ):
            return True

        # TODO: Allow this
        if (
            old_args.model_config.generation.prompt_format
            != new_args.model_config.generation.prompt_format
        ):
            return True

        return False

    @property
    def max_batch_size(self):
        return self.args.model_config.generation.max_batch_size

    @property
    def batch_wait_timeout_s(self):
        return self.args.model_config.generation.batch_wait_timeout_s

    @property
    def _ray_serve_max_batch_size(self):
        return self.max_batch_size

    @property
    def _ray_serve_batch_wait_timeout_s(self):
        return self.batch_wait_timeout_s

    async def generate_text_batch(
        self,
        prompt: Prompt,
        request: Request,
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
        **kwargs,
    ) -> AsyncGenerator:
        """Generate text from the given prompts in batch.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            request (Request): Request object.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to GATEWAY_TIMEOUT_S-10. Ignored if start_timestamp is None.
            **kwargs: Additional arguments to pass to the batch function.
        """
        curr_request_id = self.curr_request_id
        self.requests_ids[curr_request_id] = False
        self.curr_request_id += 1
        async_generator = self._generate_text_stream(
            (prompt, curr_request_id),
            start_timestamp=start_timestamp,
            timeout_s=timeout_s,
            **kwargs,
        )
        # The purpose of this loop is to ensure that the underlying
        # generator is fully consumed even if the client disconnects.
        # If the loop is not consumed, then the PredictionWorker will
        # be stuck.
        # TODO: Revisit this - consider catching asyncio.CancelledError
        # and/or setting a Ray Event to cancel the PredictionWorker generator.
        logger.info(f"self.requests_ids {self.requests_ids}")
        logger.info(f"starting request {curr_request_id}")
        while True:
            try:
                future = async_generator.__anext__()

                if not self.requests_ids[curr_request_id]:
                    future = asyncio.ensure_future(future)
                    done, pending = await asyncio.wait(
                        (future, _until_disconnected(request)),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if future in done:
                        logger.info(f"yielding request {curr_request_id}")
                        yield await future
                    else:
                        # We caught the disconnect
                        logger.info(f"Request {curr_request_id} disconnected.")
                        self.requests_ids[curr_request_id] = True
                        await future
                else:
                    logger.info(f"processing {curr_request_id}")
                    await future
            except StopAsyncIteration:
                break
        del self.requests_ids[curr_request_id]

    @serve.batch(
        # batch size & timeout will be read from
        # _ray_serve_max_batch_size and
        # _ray_serve_batch_wait_timeout_s
        batch_queue_cls=_PriorityBatchQueue,
    )
    async def _generate_text_stream(
        self,
        prompts_and_request_ids: List[Tuple[Prompt, int]],
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
        priority: QueuePriority = QueuePriority.GENERATE_TEXT,  # used by the PriorityBatchQueue
    ):
        async for result in super()._generate_text_stream(
            prompts_and_request_ids,
            start_timestamp=start_timestamp,
            timeout_s=timeout_s,
        ):
            yield result


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 2,
        "max_replicas": 8,
    },
    max_concurrent_queries=10,  # Maximum backlog for a single replica
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class ContinuousBatchingLLMDeployment(LLMDeployment):
    _predictor_cls: Type[LLMPredictor] = ContinuousBatchingPredictor

    def _should_reinit_worker_group(self, new_args: Args) -> bool:
        old_args = self.args

        if not old_args:
            return True

        old_scaling_config = self.args.air_scaling_config
        new_scaling_config = new_args.scaling_config.as_air_scaling_config()

        if not self.predictor.is_initialized():
            return True

        if old_scaling_config != new_scaling_config:
            return True

        if not old_args:
            return True

        if old_args.model_config.initialization != new_args.model_config.initialization:
            return True

        # TODO: Allow this
        if (
            old_args.model_config.generation.prompt_format
            != new_args.model_config.generation.prompt_format
        ):
            return True

        return False

    async def generate_text_batch(
        self,
        prompt: Prompt,
        request: Request,
        *,
        start_timestamp: Optional[Union[float, List[float]]] = None,
        timeout_s: Union[float, List[float]] = GATEWAY_TIMEOUT_S - 10,
        **kwargs,
    ) -> AsyncGenerator:
        """Generate text from the given prompts in batch.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            request (Request): Request object.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to GATEWAY_TIMEOUT_S-10. Ignored if start_timestamp is None.
            **kwargs: Additional arguments to pass to the batch function.
        """
        curr_request_id = self.curr_request_id
        self.requests_ids[curr_request_id] = False
        self.curr_request_id += 1
        async_generator = self._generate_text_stream(
            [(prompt, curr_request_id)],
            start_timestamp=start_timestamp,
            timeout_s=timeout_s,
            **kwargs,
        )
        try:
            async for result in async_generator:
                yield result
        finally:
            del self.requests_ids[curr_request_id]
