import asyncio
from typing import List
from unittest.mock import Mock, patch

import pytest
from starlette.requests import Request

from rayllm.backend.llm.error_handling import InputTooLong, ValidationError
from rayllm.backend.server.models import AviaryModelResponse, Prompt, QueuePriority
from rayllm.backend.server.plugins.execution_hooks import ExecutionHooks
from rayllm.backend.server.plugins.router_query_engine import StreamingErrorHandler
from rayllm.backend.server.plugins.serve_application_query_client import (
    ServeApplicationQueryClient,
)


async def mock_stream_model_responses(url, json):
    mock_stream_model_responses.args = (url, json)

    for i in range(10):
        await asyncio.sleep(0.1)
        yield AviaryModelResponse(num_generated_tokens=1, generated_text=f"t{i}, ")


async def mock_execution_hook(
    request,
    model_id: str,
    input_str: str,
    output: AviaryModelResponse,
    is_first_token: bool,
):
    mock_execution_hook.last_output = output


async def collect(async_stream):
    return [x async for x in async_stream]


@pytest.mark.asyncio
async def test_serve_application_query_engine():
    mock_request = Mock()
    mock_request.state.request_id = "request_id"

    # mock_execution_hook = AsyncMock()

    serve_engine = ServeApplicationQueryClient(
        routes={"hello": "/hello"},
        engine_configurations={"hello": Mock()},
        hooks=ExecutionHooks([mock_execution_hook]),
        metrics=Mock(),
    )

    out = None
    with patch.multiple(
        "rayllm.backend.server.plugins.serve_application_query_client",
        stream_model_responses=mock_stream_model_responses,
    ):
        out: List[AviaryModelResponse] = await collect(
            serve_engine.stream(
                model="hello",
                prompt=Prompt(prompt="hi my name is"),
                request=mock_request,
                priority=QueuePriority.GENERATE_TEXT,
            )
        )

    assert len(out) == 10, "There should be 10 response"
    for i, x in enumerate(out):
        assert x.generated_text == f"t{i}, "

    token_hook_out = mock_execution_hook.last_output
    print("Token hook out", type(token_hook_out), token_hook_out)

    merged = AviaryModelResponse.merge_stream(*out)
    assert merged.generated_text == token_hook_out.generated_text
    assert merged.num_generated_tokens == 10 == token_hook_out.num_generated_tokens

    url, json = mock_stream_model_responses.args


async def fake_generator_internal_error():
    for _ in range(4):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    raise RuntimeError("error")


async def fake_generator_pydantic_validation_error():
    for _ in range(4):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    Prompt(prompt=None)


async def fake_generator_validation_error():
    for _ in range(4):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    raise ValidationError("error")


async def fake_generator_prompt_too_long():
    for _ in range(4):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    raise InputTooLong(2, 1).exception


@pytest.fixture
def handler():
    error_handler = StreamingErrorHandler()
    request_id = "rid123"
    request = Request({"type": "http", "state": {"request_id": request_id}})
    prompt = Prompt(prompt="test")
    return error_handler, request_id, request, prompt


@pytest.mark.asyncio
async def test_streaming_error_handler_internal_server_error(handler):
    error_handler, request_id, request, prompt = handler
    generator = fake_generator_internal_error()

    async for response in error_handler.handle_failure(
        "model", request, prompt, generator
    ):
        last_response = response
    assert (
        last_response.error.message
        == f"Internal Server Error (Request ID: {request_id})"
    )
    assert (
        last_response.error.internal_message
        == f"Internal Server Error (Request ID: {request_id})"
    )
    assert last_response.error.type == "InternalServerError"
    assert last_response.error.code == 500


@pytest.mark.asyncio
async def test_streaming_error_handler_pydantic_validation_error(handler):
    error_handler, request_id, request, prompt = handler
    generator = fake_generator_pydantic_validation_error()

    async for response in error_handler.handle_failure(
        "model", request, prompt, generator
    ):
        last_response = response
    assert (
        last_response.error.message
        == f"pydantic.error_wrappers.ValidationError: 1 validation error for Prompt\nprompt\n  none is not an allowed value (type=type_error.none.not_allowed) (Request ID: {request_id})"
    )
    assert (
        last_response.error.internal_message
        == f"pydantic.error_wrappers.ValidationError: 1 validation error for Prompt\nprompt\n  none is not an allowed value (type=type_error.none.not_allowed) (Request ID: {request_id})"
    )
    assert last_response.error.type == "ValidationError"
    assert last_response.error.code == 400


@pytest.mark.asyncio
async def test_streaming_error_handler_validation_error(handler):
    error_handler, request_id, request, prompt = handler
    generator = fake_generator_validation_error()

    async for response in error_handler.handle_failure(
        "model", request, prompt, generator
    ):
        last_response = response
    assert (
        last_response.error.message
        == f"rayllm.backend.llm.error_handling.ValidationError: error (Request ID: {request_id})"
    )
    assert (
        last_response.error.internal_message
        == f"rayllm.backend.llm.error_handling.ValidationError: error (Request ID: {request_id})"
    )
    assert last_response.error.type == "ValidationError"
    assert last_response.error.code == 400


@pytest.mark.asyncio
async def test_streaming_error_handler_prompt_too_long(handler):
    error_handler, request_id, request, prompt = handler
    generator = fake_generator_prompt_too_long()

    async for response in error_handler.handle_failure(
        "model", request, prompt, generator
    ):
        last_response = response
    assert (
        last_response.error.message
        == f"rayllm.backend.llm.error_handling.PromptTooLongError: Input too long. Recieved 2 tokens, but the maximum input length is 1 tokens. (Request ID: {request_id})"
    )
    assert (
        last_response.error.internal_message
        == f"rayllm.backend.llm.error_handling.PromptTooLongError: Input too long. Recieved 2 tokens, but the maximum input length is 1 tokens. (Request ID: {request_id})"
    )
    assert last_response.error.type == "PromptTooLongError"
    assert last_response.error.code == 400
