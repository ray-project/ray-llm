import os
import time
from typing import AsyncGenerator, List

import async_timeout
from fastapi import FastAPI, status
from fastapi import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rayllm.backend.llm.vllm.vllm_models import VLLMChatCompletions, VLLMCompletions
from rayllm.backend.logger import get_logger
from rayllm.backend.observability.telemetry import configure_telemetry
from rayllm.backend.server.models import AviaryModelResponse, Prompt, QueuePriority
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.openai_compat.openai_middleware import (
    openai_exception_handler,
)
from rayllm.backend.server.plugins.router_query_engine import RouterQueryClient
from rayllm.backend.server.routers.middleware import add_request_id
from rayllm.backend.server.utils import _replace_prefix
from rayllm.common.models import (
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


# timeout in 10 minutes. Streaming can take longer than 3 min
TIMEOUT = float(os.environ.get("AVIARY_ROUTER_HTTP_TIMEOUT", 600))


def init() -> FastAPI:
    router_app = FastAPI()

    router_app.add_exception_handler(OpenAIHTTPException, openai_exception_handler)
    router_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add a unique per-request ID
    router_app.middleware("http")(add_request_id)
    # Configure common FastAPI app telemetry
    configure_telemetry(router_app, "model_router_app")

    return router_app


router_app = init()


async def _completions_wrapper(
    completion_id: str,
    body: VLLMCompletions,
    request: Request,
    response: Response,
    generator: AsyncGenerator[AviaryModelResponse, None],
) -> AsyncGenerator[str, None]:
    had_error = False
    async with async_timeout.timeout(TIMEOUT):
        all_results = []
        async for results in generator:
            for subresult in results.unpack():
                all_results.append(subresult)
                subresult_dict = subresult.dict()
                if subresult_dict.get("error"):
                    response.status_code = subresult_dict["error"]["code"]
                    # Drop finish reason as OpenAI doesn't expect it
                    # for errors in streaming
                    subresult_dict["finish_reason"] = None
                    logger.error(f"{subresult_dict['error']}")
                    all_results.pop()
                    had_error = True
                    yield "data: " + AviaryModelResponse(
                        **subresult_dict
                    ).json() + "\n\n"
                    # Return early in case of an error
                    break
                choices = [
                    TextChoice(
                        text=subresult_dict["generated_text"] or "",
                        index=0,
                        logprobs={},
                        finish_reason=subresult_dict["finish_reason"],
                    )
                ]
                usage = None
                if subresult_dict["finish_reason"]:
                    usage = (
                        Usage.from_response(
                            AviaryModelResponse.merge_stream(*all_results)
                        )
                        if all_results
                        else None
                    )
                yield "data: " + Completion(
                    id=completion_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=choices,
                    usage=usage,
                ).json() + "\n\n"
            if had_error:
                # Return early in case of an error
                break
        yield "data: [DONE]\n\n"


async def _chat_completions_wrapper(
    completion_id: str,
    body: VLLMChatCompletions,
    request: Request,
    response: Response,
    generator: AsyncGenerator[AviaryModelResponse, None],
) -> AsyncGenerator[str, None]:
    had_error = False
    async with async_timeout.timeout(TIMEOUT):
        finish_reason = None
        choices: List[DeltaChoices] = [
            DeltaChoices(
                delta=DeltaRole(role="assistant"),
                index=0,
                finish_reason=None,
            )
        ]
        yield "data: " + ChatCompletion(
            id=completion_id,
            object="text_completion",
            created=int(time.time()),
            model=body.model,
            choices=choices,
            usage=None,
        ).json() + "\n\n"

        all_results = []
        async for results in generator:
            for subresult in results.unpack():
                all_results.append(subresult)
                subresult_dict = subresult.dict()
                if subresult_dict.get("error"):
                    response.status_code = subresult_dict["error"]["code"]
                    logger.error(f"{subresult_dict['error']}")
                    # Drop finish reason as OpenAI doesn't expect it
                    # for errors in streaming
                    subresult_dict["finish_reason"] = None
                    all_results.pop()
                    had_error = True
                    yield "data: " + AviaryModelResponse(
                        **subresult_dict
                    ).json() + "\n\n"
                    # Return early in case of an error
                    break
                else:
                    finish_reason = subresult_dict["finish_reason"]
                    choices: List[DeltaChoices] = [
                        DeltaChoices(
                            delta=DeltaContent(
                                content=subresult_dict["generated_text"] or ""
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ]
                    yield "data: " + ChatCompletion(
                        id=completion_id,
                        object="text_completion",
                        created=int(time.time()),
                        model=body.model,
                        choices=choices,
                        usage=None,
                    ).json() + "\n\n"
            if had_error:
                # Return early in case of an error
                break
        if not had_error:
            choices: List[DeltaChoices] = [
                DeltaChoices(
                    delta=DeltaEOS(),
                    index=0,
                    finish_reason=finish_reason,
                )
            ]
            usage = (
                Usage.from_response(AviaryModelResponse.merge_stream(*all_results))
                if all_results
                else None
            )
            yield "data: " + ChatCompletion(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=body.model,
                choices=choices,
                usage=usage,
            ).json() + "\n\n"
        yield "data: [DONE]\n\n"


class Router:
    def __init__(
        self,
        query_engine: RouterQueryClient,
    ) -> None:
        self.query_engine = query_engine

    @router_app.get("/v1/models", response_model=Model)
    async def models(self) -> Model:
        """OpenAI API-compliant endpoint to get all Aviary models."""
        models = await self.query_engine.models()
        return Model(data=list(models.values()))

    # :path allows us to have slashes in the model name
    @router_app.get("/v1/models/{model:path}", response_model=ModelData)
    async def model_data(self, model: str) -> ModelData:
        """OpenAI API-compliant endpoint to get one Aviary model.

        :param model: The Aviary model ID (e.g. "amazon/LightGPT")
        """
        model = _replace_prefix(model)
        model_data = await self.query_engine.model(model)
        if model_data is None:
            raise OpenAIHTTPException(
                message=f"Invalid model '{model}'",
                status_code=status.HTTP_400_BAD_REQUEST,
                type="InvalidModel",
            )
        return model_data

    @router_app.post("/v1/completions")
    async def completions(
        self,
        body: VLLMCompletions,
        request: Request,
        response: FastAPIResponse,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Returns:
            A response object with completions.
        """
        prompt = Prompt(
            prompt=body.prompt,
            parameters=body,
            use_prompt_format=False,
        )

        completion_id = body.model + "-" + request.state.request_id

        if body.stream:
            return StreamingResponse(
                _completions_wrapper(
                    completion_id,
                    body,
                    request,
                    response,
                    self.query_engine.stream(
                        body.model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ),
                ),
                media_type="text/event-stream",
            )
        else:
            async with async_timeout.timeout(TIMEOUT):
                results = await self.query_engine.query(body.model, prompt, request)
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
                    id=completion_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=choices,
                    usage=usage,
                )

    @router_app.post("/v1/chat/completions")
    async def chat(
        self,
        body: VLLMChatCompletions,
        request: Request,
        response: FastAPIResponse,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Returns:
            A response object with completions.
        """
        prompt = Prompt(prompt=body.messages, parameters=body)

        completion_id = body.model + "-" + request.state.request_id

        if body.stream:
            return StreamingResponse(
                _chat_completions_wrapper(
                    completion_id,
                    body,
                    request,
                    response,
                    self.query_engine.stream(
                        body.model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ),
                ),
                media_type="text/event-stream",
            )
        else:
            async with async_timeout.timeout(TIMEOUT):
                results = await self.query_engine.query(body.model, prompt, request)
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
                    id=completion_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=choices,
                    usage=usage,
                )

    @router_app.get("/v1/health_check")
    async def health_check(self) -> bool:
        """Check if the routher is still running."""
        return True
