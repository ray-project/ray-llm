import asyncio
import json
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import aiohttp
import async_timeout
import ray
import ray.util
from fastapi import Body, FastAPI
from ray import serve
from ray.exceptions import RayActorError
from starlette.requests import Request
from starlette.responses import StreamingResponse

from aviary.backend.llm.predictor import ContinuousBatchingPredictor, LLMPredictor
from aviary.backend.logger import get_logger
from aviary.backend.server.batch import QueuePriority, _PriorityBatchQueue
from aviary.backend.server.models import (
    Args,
    DeepSpeed,
    Prompt,
    Response,
)
from aviary.common.constants import GATEWAY_TIMEOUT_S
from aviary.common.models import (
    ChatCompletion,
    Completion,
    Message,
    MessageChoices,
    Model,
    ModelData,
    TextChoice,
    Usage,
)

EOS_SENTINELS = (None, StopIteration, StopAsyncIteration)
logger = get_logger(__name__)

router_app = FastAPI()
model_app = FastAPI()


async def _until_disconnected(request: Request):
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(1)


# TODO Remove once we can stream from serve handles
@serve.ingress(model_app)
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

    # TODO Remove once we can stream from serve handles
    @model_app.get("/metadata")
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

    # TODO Remove once we can stream from serve handles
    @model_app.post("/query")
    async def generate_text(self, prompt: Prompt, request: Request) -> Response:
        return await self._generate_text(
            prompt, request, priority=QueuePriority.GENERATE_TEXT
        )

    async def _generate_text(
        self, prompt: Prompt, request: Request, *, priority: QueuePriority
    ) -> Response:
        self.predictor.validate_prompt(prompt)
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

    # TODO Remove once we can stream from serve handles
    @model_app.post("/stream")
    async def generate_text_stream(
        self, prompt: Prompt, request: Request
    ) -> StreamingResponse:
        self.predictor.validate_prompt(prompt)

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
                yield Response(
                    error="".join(traceback.format_exception_only(type(e), e)).strip(),
                    error_type=e.__class__.__name__,
                ).json() + "\n"
                raise

        return StreamingResponse(wrapper(), status_code=200, media_type="text/plain")

    # TODO Remove once we can stream from serve handles
    @model_app.post("/batch")
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
            f"Received {len(prompts)} prompts, request_ids {request_ids}. start_timestamp {start_timestamp} timeout_s {timeout_s}"
        )

        while not self.predictor.is_initialized():
            logger.info("Waiting for worker group to be initialized...")
            await asyncio.sleep(1)

        try:
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


def _replace_prefix(model: str) -> str:
    return model.replace("--", "/")


class ExecutionHooks:
    def __init__(self):
        self.hooks = []

    def add_post_execution_hook(self, fn):
        self.hooks.append(fn)

    async def trigger_post_execution_hook(
        self, request: Request, model_id: str, input_str: str, output: Response
    ):
        # Run the token hooks in parallel
        # If a token hook fails, the request will fail
        if len(self.hooks) > 0:
            await asyncio.gather(
                *[fn(request, model_id, input_str, output) for fn in self.hooks]
            )


class Router:
    def __init__(
        self,
        full_deployment_names: Dict[str, str],
        routes: Dict[str, str],
        model_configurations: Dict[str, Args],
        hooks: Optional[ExecutionHooks] = None,
    ) -> None:
        self._model_handles = {
            model_id: serve.get_deployment(deployment_name).get_handle()
            for model_id, deployment_name in full_deployment_names.items()
        }
        # TODO (shrekris-anyscale): Remove self._routes once deployments can
        # stream results to other deployments. Use Serve handles instead.
        self._routes = routes
        # TODO: Remove this once it is possible to reconfigure models on the fly
        self._model_configurations = model_configurations
        # Get the port the serve app is running on
        controller = ray.serve.context.get_global_client()._controller
        self.port = ray.get(controller.get_http_config.remote()).port
        self.hooks = hooks or ExecutionHooks()

    async def _get_response_stream(
        self, route: str, model: str, prompt: Prompt, request: Request
    ):
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(
                f"http://localhost:{self.port}{route}/stream", json=prompt.dict()
            ) as response:
                logger.info("Started receiving streaming response.")
                async for chunk in response.content:
                    pieces = chunk.split(b"\n")

                    # Track each chunk individually
                    # TODO(tchordia): we could maybe make this more efficient by combining
                    # these
                    await asyncio.gather(
                        *[
                            self.hooks.trigger_post_execution_hook(
                                request, model, prompt.prompt, Response.parse_raw(p)
                            )
                            for p in pieces
                            if p
                        ]
                    )
                    yield chunk

    @router_app.post("/stream/{model}")
    async def stream(self, model: str, prompt: Prompt, request: Request):
        model = _replace_prefix(model)
        route = self._routes[model]
        return StreamingResponse(
            self._get_response_stream(route, model, prompt, request),
            media_type="text/plain",
        )

    @router_app.post("/query/{model}")
    async def query(
        self, model: str, prompt: Prompt, request: Request
    ) -> Dict[str, Any]:
        return (await self.batch_query(model, [prompt], request))[0]

    @router_app.post("/query/batch/{model}")
    async def batch_query(
        self, model: str, prompts: List[Prompt], request: Request
    ) -> List[Dict[str, Any]]:
        model = _replace_prefix(model)
        route = self._routes[model]
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(
                f"http://localhost:{self.port}{route}/batch",
                json=[p.dict() for p in prompts],
            ) as response:
                logger.info(
                    "Received response from intermediate request. Awaiting json body."
                )
                completions = await response.json()

                # Set execution state on the request object for middlewares
                for completion, prompt in zip(completions, prompts):
                    await self.hooks.trigger_post_execution_hook(
                        request, model, prompt.prompt, Response.parse_obj(completion)
                    )

                return completions

    @router_app.get("/metadata/{model}")
    async def metadata(self, model) -> Dict[str, Dict[str, Any]]:
        model = _replace_prefix(model)
        # This is what we want to do eventually, but it looks like reconfigure
        # is blocking when called on replica init
        # metadata = await asyncio.gather(
        #     *(await asyncio.gather(*[self._models[model].metadata.remote()]))
        # )
        # metadata = metadata[0]
        metadata = self._model_configurations[model].dict(
            exclude={
                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
            }
        )
        logger.info(metadata)
        return {"metadata": metadata}

    @router_app.get("/models")
    async def models_v0(self) -> List[str]:
        return list(self._model_handles.keys())

    @router_app.get("/v1/models", response_model=Model)
    async def models(self) -> Model:
        """OpenAI API-compliant endpoint to get all Aviary models."""
        model_ids = list(self._model_handles.keys())
        model_data = []
        for model_id in model_ids:
            model_data.append(
                ModelData(
                    id=model_id,
                    object="model",
                    owned_by="organization-owner",  # TODO: define owner (metadata)
                    permission=[],  # TODO: define permissions (metadata)
                )
            )
        return Model(data=model_data)

    @router_app.get("/v1/models/{model}", response_model=ModelData)
    async def model_data(self, model: str) -> ModelData:
        """OpenAI API-compliant endpoint to get one Aviary model.

        :param model: The Aviary model ID (e.g. "amazon/LightGPT")
        """
        # TODO: should we integrate "metadata" here?
        return ModelData(
            id=model,
            object="model",
            owned_by="organization-owner",  # TODO
            permission=[],  # TODO
        )

    @router_app.post("/v1/completions/{model}")
    async def completions(
        self,
        model: str,
        prompt: Annotated[str, Body()],
        request: Request,
        suffix: Annotated[Optional[str], Body()] = None,
        max_tokens: Annotated[int, Body()] = 32,
        temperature: Annotated[float, Body()] = 1.0,
        top_p: Annotated[float, Body()] = 1.0,
        n: Annotated[int, Body()] = 1,
        stream: Annotated[bool, Body()] = False,
        logprobs: Annotated[Optional[int], Body()] = None,
        echo: Annotated[bool, Body()] = False,
        stop: Annotated[Optional[List[str]], Body()] = None,
        presence_penalty: Annotated[float, Body()] = 0.0,
        frequency_penalty: Annotated[float, Body()] = 0.0,
        best_of: Annotated[int, Body()] = 1,
        logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
        user: Annotated[Optional[str], Body()] = None,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Args:
            model: The model to query.
            prompt: The prompt to generate completions for, encoded as string.
            suffix: The suffix that comes after a completion of inserted text.
            max_tokens: The maximum number of tokens to generate.
            temperature: What sampling temperature to use.
            top_p: An alternative to sampling with temperature, called nucleus sampling.
            n: How many completions to generate for each prompt.
            stream: Whether to stream back partial progress.
            logprobs: Include the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens.
            echo: Echo back the prompt in addition to the completion.
            stop: Up to 4 sequences where the API will stop generating further tokens.
                The returned text will not contain the stop sequence.
            presence_penalty: Number between -2.0 and 2.0.
                Positive values penalize new tokens based on whether they appear in
                the text so far, increasing the model's likelihood to talk about
                new topics.
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            best_of: Generates `best_of` completions server-side and returns the "best".
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.

        Returns:
            A response object with completions.
        """
        prompt = Prompt(
            prompt=prompt,
            parameters={
                "temperature": temperature,
                "do_sample": temperature > 0,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "repetition_penalty": frequency_penalty,
            },
            stopping_sequences=stop,
            use_prompt_format=False,
        )
        if stream:
            model = _replace_prefix(model)
            route = self._routes[model]

            async def completions_wrapper():
                async for response in self._get_response_stream(
                    route, model, prompt, request
                ):
                    response = json.loads(response)
                    if response.get("error"):
                        yield Response(**results).json() + "\n"
                    else:
                        choices = [
                            TextChoice(
                                text=response["generated_text"],
                                index=0,
                                logprobs={},
                                finish_reason="length",
                            )
                        ]
                        usage = Usage.from_response(response)
                        yield Completion(
                            id=model + "-" + str(uuid.uuid4()),
                            object="text_completion",
                            created=int(time.time()),
                            model=model,
                            choices=choices,
                            usage=usage,
                        ).json() + "\n"

            return StreamingResponse(
                completions_wrapper(),
                media_type="text/plain",
            )
        else:
            results = await self.query(model, prompt, request)
            if results.get("error"):
                return Response(**results)

            choices = [
                TextChoice(
                    text=results["generated_text"],
                    index=0,
                    logprobs={},
                    finish_reason="length",
                )
            ]
            usage = Usage.from_response(results)
            # TODO: pick up parameters that make sense, remove the rest

            return Completion(
                id=model + "-" + str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=choices,
                usage=usage,
            )

    @router_app.post("/v1/chat/completions/{model}")
    async def chat(
        self,
        model: str,
        messages: List[Message],
        request: Request,
        temperature: Annotated[float, Body()] = 1.0,
        top_p: Annotated[float, Body()] = 1.0,
        n: Annotated[int, Body()] = 1,
        stream: Annotated[bool, Body()] = False,
        logprobs: Annotated[Optional[int], Body()] = None,
        echo: Annotated[bool, Body()] = False,
        stop: Annotated[Optional[List[str]], Body()] = None,
        presence_penalty: Annotated[float, Body()] = 0.0,
        frequency_penalty: Annotated[float, Body()] = 0.0,
        logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
        user: Annotated[Optional[str], Body()] = None,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Args:
            model: The model to query.
            messages: A list of messages describing the conversation so far.
                Contains a required "role", which is the role of the author of this
                message. One of "system", "user", or "assistant".
                Also contains required "content", the contents of the message, and
                an optional "name", the name of the author of this message.
            temperature: What sampling temperature to use.
            top_p: An alternative to sampling with temperature, called nucleus sampling.
            n: How many completions to generate for each prompt.
            stream: Whether to stream back partial progress.
            logprobs: Include the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens.
            echo: Echo back the prompt in addition to the completion.
            stop: Up to 4 sequences where the API will stop generating further tokens.
                The returned text will not contain the stop sequence.
            presence_penalty: Number between -2.0 and 2.0.
                Positive values penalize new tokens based on whether they appear in
                the text so far, increasing the model's likelihood to talk about
                new topics.
            frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
                new tokens based on their existing frequency in the text so far,
                decreasing the model's likelihood to repeat the same line verbatim.
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.

        Returns:
            A response object with completions.
        """
        prompt = Prompt(
            prompt=messages,
            parameters={
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": top_p,
                "repetition_penalty": frequency_penalty,
            },
            stopping_sequences=stop,
        )

        if stream:
            model = _replace_prefix(model)
            route = self._routes[model]

            async def completions_wrapper():
                async for response in self._get_response_stream(
                    route, model, prompt, request
                ):
                    response = json.loads(response)
                    if response.get("error"):
                        yield Response(**results).json() + "\n"
                    else:
                        choices: List[MessageChoices] = [
                            MessageChoices(
                                message=Message(
                                    role="assistant", content=response["generated_text"]
                                ),
                                index=0,
                                finish_reason="length",
                            )
                        ]
                        usage = Usage.from_response(response)
                        yield ChatCompletion(
                            id=model + "-" + str(uuid.uuid4()),
                            object="text_completion",
                            created=int(time.time()),
                            model=model,
                            choices=choices,
                            usage=usage,
                        ).json() + "\n"

            return StreamingResponse(
                completions_wrapper(),
                media_type="text/plain",
            )
        else:
            results = await self.query(model, prompt, request)
            if results.get("error"):
                return Response(**results)

            # TODO: pick up parameters that make sense, remove the rest

            choices: List[MessageChoices] = [
                MessageChoices(
                    message=Message(
                        role="assistant", content=results["generated_text"]
                    ),
                    index=0,
                    finish_reason="length",
                )
            ]
            usage = Usage.from_response(results)

            return ChatCompletion(
                id=model + "-" + str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=choices,
                usage=usage,
            )


RouterDeployment = serve.deployment(
    route_prefix="/",
    # TODO make this configurable in aviary run
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 16,
        "target_num_ongoing_requests_per_replica": 100,
    },
    max_concurrent_queries=500,  # Maximum backlog for a single replica
)(serve.ingress(router_app)(Router))
