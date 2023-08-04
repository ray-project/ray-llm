import os
import time
import uuid
from typing import Dict, List, Optional

from aviary.backend.server.execution_hooks import ExecutionHooks
from aviary.backend.server.metrics import Metrics
from aviary.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from aviary.backend.server.openai_compat.openai_middleware import (
    openai_exception_handler,
)
from aviary.backend.server.utils import _replace_prefix

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

import aiohttp
import async_timeout
import ray
import ray.util
from fastapi import Body, FastAPI, status
from fastapi import Response as FastAPIResponse
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse

from aviary.backend.logger import get_logger
from aviary.backend.server.models import (
    Args,
    AviaryModelResponse,
    Prompt,
)
from aviary.backend.server.utils import QueuePriority
from aviary.common.models import (
    ChatCompletion,
    Completion,
    DeltaChoices,
    DeltaContent,
    DeltaEOS,
    DeltaRole,
    Message,
    MessageChoices,
    Model,
    ModelData,
    TextChoice,
    Usage,
)

logger = get_logger(__name__)

router_app = FastAPI()
router_app.add_exception_handler(OpenAIHTTPException, openai_exception_handler)
TIMEOUT = float(os.environ.get("AVIARY_ROUTER_HTTP_TIMEOUT", 175))


class Router:
    def __init__(
        self,
        full_deployment_names: Dict[str, str],
        routes: Dict[str, str],
        engine_configurations: Dict[str, Args],
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
        self._engine_configurations = engine_configurations
        # Get the port the serve app is running on
        controller = ray.serve.context.get_global_client()._controller
        self.port = ray.get(controller.get_http_config.remote()).port
        self.hooks = hooks or ExecutionHooks()
        self.metrics = Metrics()

    async def _get_response_stream(
        self,
        route: str,
        model: str,
        prompt: Prompt,
        request: Request,
        priority: QueuePriority,
    ):
        is_first_token = True
        model_tags = {"model_id": model}
        self.metrics.requests_started.inc(tags=model_tags)

        req_id = uuid.uuid4().hex
        start_time = time.monotonic()
        logger.info(f"Starting request handling. Id: {req_id}.")

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(
                f"http://localhost:{self.port}{route}/stream",
                json={
                    "prompt": prompt.dict(),
                    "priority": int(priority),
                },
                timeout=TIMEOUT,
            ) as response:
                async for chunk in response.content.iter_any():
                    responses = [
                        AviaryModelResponse.parse_raw(p)
                        for p in chunk.split(b"\n")
                        if p
                    ]
                    combined_response = AviaryModelResponse.merge_stream(*responses)

                    await self.hooks.trigger_post_execution_hook(
                        request, model, prompt.prompt, is_first_token, combined_response
                    )

                    # Track metrics
                    for res in responses:
                        self.metrics.track(res, is_first_token, model)
                        is_first_token = False
                        yield res
        end_time = time.monotonic()
        logger.info(
            f"Completed request handling. Id: {req_id}. Took {end_time - start_time}s"
        )

    async def _query(self, model: str, prompt: Prompt, request: Request):
        route = self._get_model_path(model)
        response_stream = self._get_response_stream(
            route,
            model,
            prompt,
            request,
            priority=QueuePriority.BATCH_GENERATE_TEXT,
        )
        responses = [resp async for resp in response_stream]
        return AviaryModelResponse.merge_stream(*responses)

    @router_app.get("/v1/models", response_model=Model)
    async def models(self) -> Model:
        """OpenAI API-compliant endpoint to get all Aviary models."""
        model_ids = list(self._model_handles.keys())
        return Model(data=[self._model(model_id) for model_id in model_ids])

    # :path allows us to have slashes in the model name
    @router_app.get("/v1/models/{model:path}", response_model=ModelData)
    async def model_data(self, model: str) -> ModelData:
        """OpenAI API-compliant endpoint to get one Aviary model.

        :param model: The Aviary model ID (e.g. "amazon/LightGPT")
        """
        model = _replace_prefix(model)
        return self._model(model)

    @router_app.post("/v1/completions")
    async def completions(
        self,
        model: Annotated[str, Body()],
        prompt: Annotated[str, Body()],
        request: Request,
        response: FastAPIResponse,
        suffix: Annotated[Optional[str], Body()] = None,
        max_tokens: Annotated[Optional[int], Body()] = 16,
        temperature: Annotated[Optional[float], Body()] = None,
        top_p: Annotated[Optional[float], Body()] = None,
        n: Annotated[int, Body()] = 1,
        stream: Annotated[bool, Body()] = False,
        logprobs: Annotated[Optional[int], Body()] = None,
        echo: Annotated[bool, Body()] = False,
        stop: Annotated[Optional[List[str]], Body()] = None,
        presence_penalty: Annotated[float, Body()] = None,
        frequency_penalty: Annotated[float, Body()] = None,
        best_of: Annotated[int, Body()] = 1,
        logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
        user: Annotated[Optional[str], Body()] = None,
        top_k: Annotated[Optional[int], Body()] = None,
        typical_p: Annotated[Optional[float], Body()] = None,
        watermark: Annotated[Optional[bool], Body()] = False,
        seed: Annotated[Optional[int], Body()] = None,
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
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            typical_p: Typical Decoding mass. See Typical Decoding for Natural Language Generation
                (https://arxiv.org/abs/2202.00666) for more information.
            watermark: Watermarking with A Watermark for Large Language Models
                (https://arxiv.org/abs/2301.10226).
            seed: Random sampling seed.


        Returns:
            A response object with completions.
        """
        prompt = Prompt(
            prompt=prompt,
            parameters={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "top_k": top_k,
                "typical_p": typical_p,
                "watermark": watermark,
                "seed": seed,
            },
            stopping_sequences=stop,
            use_prompt_format=False,
        )

        route = self._get_model_path(model)

        if stream:

            async def completions_wrapper():
                with async_timeout.timeout(TIMEOUT):
                    async for results in self._get_response_stream(
                        route,
                        model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ):
                        results = results.dict()
                        if results.get("error"):
                            response.status_code = results["error"]["code"]
                            logger.error(f"{results['error']}")
                            yield AviaryModelResponse(**results).json() + "\n"
                        elif not results["finish_reason"]:
                            choices = [
                                TextChoice(
                                    text=results["generated_text"] or "",
                                    index=0,
                                    logprobs={},
                                    finish_reason=None,
                                )
                            ]
                            yield "data: " + Completion(
                                id=model + "-" + str(uuid.uuid4()),
                                object="text_completion",
                                created=int(time.time()),
                                model=model,
                                choices=choices,
                                usage=None,
                            ).json() + "\n"
                    yield "data: [DONE]\n"

            return StreamingResponse(
                completions_wrapper(),
                media_type="text/event-stream",
            )
        else:
            with async_timeout.timeout(TIMEOUT):
                results = await self._query(model, prompt, request)
                if results.error:
                    raise OpenAIHTTPException(
                        message=results.error.message,
                        status_code=results.error.code,
                        type=results.error.type,
                    )
                results = results.dict()

                choices = [
                    TextChoice(
                        text=results["generated_text"] or "",
                        index=0,
                        logprobs={},
                        finish_reason=results["finish_reason"],
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

    @router_app.post("/v1/chat/completions")
    async def chat(
        self,
        model: Annotated[str, Body()],
        messages: List[Message],
        request: Request,
        response: FastAPIResponse,
        max_tokens: Annotated[Optional[int], Body()] = None,
        temperature: Annotated[Optional[float], Body()] = None,
        top_p: Annotated[Optional[float], Body()] = None,
        n: Annotated[int, Body()] = 1,
        stream: Annotated[bool, Body()] = False,
        logprobs: Annotated[Optional[int], Body()] = None,
        echo: Annotated[bool, Body()] = False,
        stop: Annotated[Optional[List[str]], Body()] = None,
        presence_penalty: Annotated[float, Body()] = None,
        frequency_penalty: Annotated[float, Body()] = None,
        logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
        user: Annotated[Optional[str], Body()] = None,
        top_k: Annotated[Optional[int], Body()] = None,
        typical_p: Annotated[Optional[float], Body()] = None,
        watermark: Annotated[Optional[bool], Body()] = False,
        seed: Annotated[Optional[int], Body()] = None,
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
            max_tokens: The maximum number of tokens to generate. Defaults to inf.
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
            best_of: Generates `best_of` completions server-side and returns the "best".
            logit_bias: Modify the likelihood of specified tokens appearing in
                the completion.
            user: A unique identifier representing your end-user, which can help us
                to monitor and detect abuse. Learn more.
            top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
            typical_p: Typical Decoding mass. See Typical Decoding for Natural Language Generation
                (https://arxiv.org/abs/2202.00666) for more information.
            watermark: Watermarking with A Watermark for Large Language Models
                (https://arxiv.org/abs/2301.10226).
            seed: Random sampling seed.

        Returns:
            A response object with completions.
        """
        print(
            {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "top_k": top_k,
                "typical_p": typical_p,
                "watermark": watermark,
                "seed": seed,
            }
        )
        prompt = Prompt(
            prompt=messages,
            parameters={
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "top_k": top_k,
                "typical_p": typical_p,
                "watermark": watermark,
                "seed": seed,
            },
            stopping_sequences=stop,
        )
        route = self._get_model_path(model)

        if stream:

            async def completions_wrapper():
                with async_timeout.timeout(TIMEOUT):
                    finish_reason = None
                    choices: List[DeltaChoices] = [
                        DeltaChoices(
                            delta=DeltaRole(role="assistant"),
                            index=0,
                            finish_reason=None,
                        )
                    ]
                    yield "data: " + ChatCompletion(
                        id=model + "-" + str(uuid.uuid4()),
                        object="text_completion",
                        created=int(time.time()),
                        model=model,
                        choices=choices,
                        usage=None,
                    ).json() + "\n"

                    all_results = []
                    async for results in self._get_response_stream(
                        route,
                        model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ):
                        results = results.dict()
                        if results.get("error"):
                            response.status_code = results["error"]["code"]
                            logger.error(f"{results['error']}")
                            yield "data: " + AviaryModelResponse(
                                **results
                            ).json() + "\n"
                        else:
                            all_results.append(AviaryModelResponse(**results))
                            finish_reason = results["finish_reason"]
                            if finish_reason:
                                continue
                            choices: List[DeltaChoices] = [
                                DeltaChoices(
                                    delta=DeltaContent(
                                        content=results["generated_text"] or ""
                                    ),
                                    index=0,
                                    finish_reason=None,
                                )
                            ]
                            yield "data: " + ChatCompletion(
                                id=model + "-" + str(uuid.uuid4()),
                                object="text_completion",
                                created=int(time.time()),
                                model=model,
                                choices=choices,
                                usage=None,
                            ).json() + "\n"
                    choices: List[DeltaChoices] = [
                        DeltaChoices(
                            delta=DeltaEOS(),
                            index=0,
                            finish_reason=finish_reason,
                        )
                    ]
                    usage = (
                        Usage.from_response(
                            AviaryModelResponse.merge_stream(*all_results)
                        )
                        if all_results
                        else None
                    )
                    yield "data: " + ChatCompletion(
                        id=model + "-" + str(uuid.uuid4()),
                        object="text_completion",
                        created=int(time.time()),
                        model=model,
                        choices=choices,
                        usage=usage,
                    ).json() + "\n"
                    yield "data: [DONE]\n"

            return StreamingResponse(
                completions_wrapper(),
                media_type="text/event-stream",
            )
        else:
            with async_timeout.timeout(TIMEOUT):
                results = await self._query(model, prompt, request)
                if results.error:
                    raise OpenAIHTTPException(
                        message=results.error.message,
                        status_code=results.error.code,
                        type=results.error.type,
                    )
                results = results.dict()

                # TODO: pick up parameters that make sense, remove the rest

                choices: List[MessageChoices] = [
                    MessageChoices(
                        message=Message(
                            role="assistant", content=results["generated_text"] or ""
                        ),
                        index=0,
                        finish_reason=results["finish_reason"],
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

    @router_app.get("/v1/health_check")
    async def health_check(self) -> bool:
        """Check if the routher is still running."""
        return True

    def _model(self, model: str):
        metadata = self._engine_configurations[model].dict(
            include={
                "engine_config": {
                    "generation",
                    "model_id",
                    "model_url",
                    "model_description",
                }
            }
        )
        return ModelData(
            id=model,
            object="model",
            owned_by="organization-owner",  # TODO
            permission=[],  # TODO
            aviary_metadata=metadata,
        )

    def _get_model_path(self, model: str):
        model = _replace_prefix(model)
        route = self._routes.get(model)
        if route is None:
            raise OpenAIHTTPException(
                message=f"Invalid model '{model}'",
                status_code=status.HTTP_400_BAD_REQUEST,
                type="InvalidModel",
            )
        return route
