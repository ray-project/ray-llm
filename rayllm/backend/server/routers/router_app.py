import asyncio
import os
import time
from typing import AsyncGenerator, List, Optional, Tuple

import async_timeout
from fastapi import FastAPI, HTTPException, status
from fastapi import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from httpx import HTTPStatusError as HTTPXHTTPStatusError
from ray import serve
from starlette.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from rayllm.backend.llm.embedding.embedding_models import Embeddings
from rayllm.backend.logger import get_logger
from rayllm.backend.observability.telemetry import configure_telemetry
from rayllm.backend.server.models import (
    AviaryModelResponse,
    ChatCompletionsParams,
    CompletionsParams,
    Prompt,
    QueuePriority,
)
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.openai_compat.openai_middleware import (
    openai_exception_handler,
)
from rayllm.backend.server.plugins.router_query_engine import RouterQueryClient
from rayllm.backend.server.routers.middleware import add_request_id
from rayllm.backend.server.utils import _replace_prefix, get_response_for_error
from rayllm.common.models import (
    ChatCompletion,
    ChoiceLogProbs,
    Completion,
    DeletedModel,
    DeltaChoices,
    DeltaContent,
    DeltaEOS,
    DeltaRole,
    EmbeddingsData,
    EmbeddingsOutput,
    EmbeddingsUsage,
    LogProbs,
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
    router_app.add_exception_handler(HTTPException, openai_exception_handler)
    router_app.add_exception_handler(HTTPXHTTPStatusError, openai_exception_handler)
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
    # this is necessary for passing through exceptions to users,
    # seems to be some flaws of starlette, see discussion at
    # https://github.com/encode/starlette/issues/1175
    router_app.add_middleware(
        ExceptionMiddleware, handlers=router_app.exception_handlers
    )

    return router_app


router_app = init()


async def _openai_json_generator(
    generator: AsyncGenerator[AviaryModelResponse, None],
    first_response: Optional[AviaryModelResponse] = None,
):
    if first_response is not None:
        yield "data: " + first_response.json() + "\n\n"
    async for response in generator:
        yield "data: " + response.json() + "\n\n"
    yield "data: [DONE]\n\n"


async def _peek_at_openai_json_generator(
    generator: AsyncGenerator[AviaryModelResponse, None]
) -> Tuple[AviaryModelResponse, AsyncGenerator[str, None]]:
    """Runs one iteration of the underlying generator
    and returns the result alongside the generator itself (with the
    first iteration still there).
    """
    first_response = await generator.__anext__()
    return first_response, _openai_json_generator(generator, first_response)


async def _completions_wrapper(
    model: str,
    request_id: str,
    response: Response,
    generator: AsyncGenerator[AviaryModelResponse, None],
) -> AsyncGenerator[AviaryModelResponse, None]:
    had_error = False
    completion_id = _get_model_request_id(model, request_id)
    async with async_timeout.timeout(TIMEOUT):
        all_results = []
        try:
            async for results in generator:
                for subresult in results.unpack():
                    all_results.append(subresult)
                    subresult_dict = subresult.dict()
                    if subresult_dict.get("error"):
                        response.status_code = subresult_dict["error"]["code"]
                        # Drop finish reason as OpenAI doesn't expect it
                        # for errors in streaming
                        subresult_dict["finish_reason"] = None
                        logger.error(
                            f"Reporting back an error: {subresult_dict['error']}"
                        )
                        all_results.pop()
                        had_error = True
                        yield AviaryModelResponse.parse_obj(subresult_dict)
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
                    yield Completion(
                        id=completion_id,
                        object="text_completion",
                        created=int(time.time()),
                        model=model,
                        choices=choices,
                        usage=usage,
                    )
                if had_error:
                    # Return early in case of an error
                    break
        except Exception as e:
            logger.error(
                f"Failed while handling completions for request ({request_id}): {repr(e)}",
                exc_info=e,
            )

            exc_response = get_response_for_error(e, request_id)
            response.status_code = exc_response.error.code
            had_error = True
            yield exc_response


async def _chat_completions_wrapper(
    model: str,
    request_id: str,
    response: Response,
    generator: AsyncGenerator[AviaryModelResponse, None],
) -> AsyncGenerator[AviaryModelResponse, None]:
    had_error = False
    completion_id = _get_model_request_id(model, request_id)
    async with async_timeout.timeout(TIMEOUT):
        finish_reason = None
        choices: List[DeltaChoices] = [
            DeltaChoices(
                delta=DeltaRole(role="assistant"),
                index=0,
                finish_reason=None,
            )
        ]

        yielded_role = False
        all_results = []
        try:
            async for results in generator:
                for subresult in results.unpack():
                    logger.info(f"subresult: {subresult}")
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
                        yield AviaryModelResponse.parse_obj(subresult_dict)
                        # Return early in case of an error
                        break
                    else:
                        finish_reason = subresult_dict["finish_reason"]

                        if not yielded_role:
                            choices: List[DeltaChoices] = [
                                DeltaChoices(
                                    delta=DeltaRole(role="assistant"),
                                    index=0,
                                    finish_reason=None,
                                    logprobs=ChoiceLogProbs(content=[]),
                                )
                            ]
                            yield ChatCompletion(
                                id=completion_id,
                                object="text_completion",
                                created=int(time.time()),
                                model=model,
                                choices=choices,
                                usage=None,
                            )
                            yielded_role = True
                        if subresult_dict["logprobs"]:
                            logprobs = ChoiceLogProbs(
                                content=[
                                    LogProbs.parse_obj(logprob)
                                    for logprob in subresult_dict["logprobs"]
                                ]
                            )
                        else:
                            logprobs = None
                        choices: List[DeltaChoices] = [
                            DeltaChoices(
                                delta=DeltaContent(
                                    content=subresult_dict["generated_text"] or "",
                                    tool_calls=subresult_dict["tool_calls"] or None,
                                ),
                                index=0,
                                finish_reason=None,
                                logprobs=logprobs,
                            )
                        ]
                        yield ChatCompletion(
                            id=completion_id,
                            object="text_completion",
                            created=int(time.time()),
                            model=model,
                            choices=choices,
                            usage=None,
                        )
                if had_error:
                    # Return early in case of an error
                    break
        except Exception as e:
            logger.error(
                f"Failed while handling chat-completions for request ({request_id}): {repr(e)}",
                exc_info=e,
            )

            exc_response = get_response_for_error(e, request_id)
            response.status_code = exc_response.error.code
            had_error = True
            yield exc_response

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
            yield ChatCompletion(
                id=completion_id,
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=choices,
                usage=usage,
            )


class Router:
    def __init__(
        self,
        query_engine: RouterQueryClient,
    ) -> None:
        # Increase the amount of time allocated for fetching the queue length
        # TODO(tchordia): use the associated env var instead once it's available
        serve._private.router.PowerOfTwoChoicesReplicaScheduler.queue_len_response_deadline_s = (
            0.5
        )
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

    @router_app.delete("/v1/models/{model:path}", response_model=DeletedModel)
    async def delete_fine_tuned_model(self, model: str) -> DeletedModel:
        """OpenAI API-compliant endpoint to delete one fine-tuned model.

        :param model: The fine-tuned model ID (e.g. "meta-llama/Llama-2-7b-chat-hf:john:aBc1234")
        """
        model = _replace_prefix(model)
        await self.query_engine.delete_fine_tuned_model(model)
        return DeletedModel(id=model)

    @router_app.post("/v1/completions")
    async def completions(
        self,
        body: CompletionsParams,
        request: Request,
        response: FastAPIResponse,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Returns:
            A response object with completions.
        """
        req_id = request.state.request_id
        prompt = Prompt(
            prompt=body.prompt,
            parameters=body,
            use_prompt_format=False,
        )

        if body.stream:
            first_response, wrapper = await _peek_at_openai_json_generator(
                _completions_wrapper(
                    body.model,
                    req_id,
                    response,
                    self.query_engine.stream(
                        body.model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ),
                ),
            )
            if isinstance(first_response, AviaryModelResponse) and first_response.error:
                raise OpenAIHTTPException.from_model_response(first_response)
            return StreamingResponse(wrapper, media_type="text/event-stream")
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
                    id=_get_model_request_id(body.model, req_id),
                    object="text_completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=choices,
                    usage=usage,
                )

    @router_app.post("/v1/chat/completions")
    async def chat(
        self,
        body: ChatCompletionsParams,
        request: Request,
        response: FastAPIResponse,
    ):
        """Given a prompt, the model will return one or more predicted completions,
        and can also return the probabilities of alternative tokens at each position.

        Returns:
            A response object with completions.
        """
        tools = body.tools
        tool_choice = body.tool_choice
        # Doing this to remove them from sampling params
        body.tools = None
        body.tool_choice = None

        req_id = request.state.request_id
        prompt = Prompt(
            prompt=body.messages, parameters=body, tools=tools, tool_choice=tool_choice
        )

        if body.stream:
            first_response, wrapper = await _peek_at_openai_json_generator(
                _chat_completions_wrapper(
                    body.model,
                    req_id,
                    response,
                    self.query_engine.stream(
                        body.model,
                        prompt,
                        request,
                        priority=QueuePriority.GENERATE_TEXT,
                    ),
                ),
            )
            if isinstance(first_response, AviaryModelResponse) and first_response.error:
                raise OpenAIHTTPException.from_model_response(first_response)
            return StreamingResponse(wrapper, media_type="text/event-stream")
        else:
            async with async_timeout.timeout(TIMEOUT):
                results = await self.query_engine.query(body.model, prompt, request)
                if results.error:
                    raise OpenAIHTTPException(
                        message=results.error.message,
                        status_code=results.error.code,
                        type=results.error.type,
                    )
                # TODO: pick up parameters that make sense, remove the rest
                logprobs = results.logprobs
                if logprobs:
                    logprobs = ChoiceLogProbs(
                        content=[LogProbs.parse_obj(logprob) for logprob in logprobs]
                    )
                else:
                    logprobs = None
                if results.tool_calls:
                    msg = Message(role="assistant", tool_calls=results.tool_calls)
                    # deleting this fields so that they don't appear in the response
                    del msg.tool_call_id
                    choices: List[MessageChoices] = [
                        MessageChoices(
                            message=msg,
                            index=0,
                            finish_reason=results.finish_reason,
                            logprobs=logprobs,
                        )
                    ]
                else:
                    choices: List[MessageChoices] = [
                        MessageChoices(
                            message=Message(
                                role="assistant",
                                content=results.generated_text or "",
                            ),
                            index=0,
                            finish_reason=results.finish_reason,
                            logprobs=logprobs,
                        )
                    ]

                usage = Usage.from_response(results)

                return ChatCompletion(
                    id=_get_model_request_id(body.model, req_id),
                    object="text_completion",
                    created=int(time.time()),
                    model=body.model,
                    choices=choices,
                    usage=usage,
                )

    @router_app.post("/v1/embeddings")
    async def embed(
        self,
        body: Embeddings,
        request: Request,
    ):
        """Given a prompt, the model will return one embedding.

        Returns:
            A response object with an embedding.
        """
        embedding_id = _get_model_request_id(body.model, request.state.request_id)

        async with async_timeout.timeout(TIMEOUT):
            if isinstance(body.input, str):
                input = [body.input]
            else:
                input = body.input
            prompts = [Prompt(prompt=x, parameters=body) for x in input]
            results_list: List[AviaryModelResponse] = await asyncio.gather(
                *[
                    self.query_engine.query(body.model, prompt, request)
                    for prompt in prompts
                ]
            )
            final_results = []
            tokens = 0
            for results in results_list:
                if results.error:
                    raise OpenAIHTTPException.from_model_response(results)
                final_results.append(results.dict())
                tokens += results.num_input_tokens

            return EmbeddingsOutput(
                data=[
                    EmbeddingsData(
                        embedding=results["embedding_outputs"],
                        index=i,
                        object="embedding",
                    )
                    for i, results in enumerate(final_results)
                ],
                id=embedding_id,
                object="list",
                created=int(time.time()),
                model=body.model,
                usage=EmbeddingsUsage(
                    prompt_tokens=tokens,
                    total_tokens=tokens,
                ),
            )

    @router_app.get("/v1/health_check")
    async def health_check(self) -> bool:
        """Check if the routher is still running."""
        return True


def _get_model_request_id(model: str, request_id: str):
    return model + "-" + request_id
