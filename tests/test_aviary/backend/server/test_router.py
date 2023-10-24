import json

import pytest
from starlette.requests import Request
from starlette.responses import Response

from rayllm.backend.llm.vllm.vllm_models import VLLMChatCompletions, VLLMCompletions
from rayllm.backend.server.models import (
    AviaryModelResponse,
    BatchedAviaryModelResponse,
    ErrorResponse,
    Message,
)
from rayllm.backend.server.routers.router_app import (
    _chat_completions_wrapper,
    _completions_wrapper,
)


async def fake_generator():
    for _ in range(10):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield AviaryModelResponse(
        num_generated_tokens=1, generated_text="abcd", finish_reason="stop"
    )


async def fake_generator_with_error():
    for _ in range(5):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield AviaryModelResponse(
        error=ErrorResponse(
            message="error", internal_message="error", code=500, type="error"
        ),
        finish_reason="error",
    )


async def fake_generator_batched():
    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
    )
    yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
    )
    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
    )
    yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(
            num_generated_tokens=1, generated_text="abcd", finish_reason="stop"
        ),
    )


async def fake_generator_with_error_batched():
    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
    )
    yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")

    yield BatchedAviaryModelResponse.merge_stream(
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(num_generated_tokens=1, generated_text="abcd"),
        AviaryModelResponse(
            error=ErrorResponse(
                message="error", internal_message="error", code=500, type="error"
            ),
            finish_reason="error",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [fake_generator, fake_generator_batched])
async def test_completions_stream(generator):
    generator = _completions_wrapper(
        "1",
        VLLMCompletions(model="test", prompt="test"),
        Request({"type": "http"}, {"request_id": "1"}),
        Response(),
        generator(),
    )
    count = 0
    had_done = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            assert "id" in dct
            assert "object" in dct
            assert "choices" in dct
            assert dct["choices"][0]["text"] == "abcd"
    assert dct["choices"][0]["finish_reason"] == "stop"
    assert had_done
    assert count == 12
    assert dct["usage"]["completion_tokens"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generator", [fake_generator_with_error, fake_generator_with_error_batched]
)
async def test_completions_stream_with_error(generator):
    generator = _completions_wrapper(
        "1",
        VLLMCompletions(model="test", prompt="test"),
        Request({"type": "http"}, {"request_id": "1"}),
        Response(),
        generator(),
    )
    count = 0
    had_error = False
    had_done = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            if "error" in dct:
                had_error = True
                assert dct["error"]["message"] == "error"
            elif had_error:
                raise AssertionError()
            else:
                assert "id" in dct
                assert "object" in dct
                assert "choices" in dct
                assert dct["choices"][0]["text"] == "abcd"
    assert dct["error"]
    assert had_error
    assert count == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("generator", [fake_generator, fake_generator_batched])
async def test_chat_completions_stream(generator):
    generator = _chat_completions_wrapper(
        "1",
        VLLMChatCompletions(
            model="test", messages=[Message(role="user", content="test")]
        ),
        Request({"type": "http"}, {"request_id": "1"}),
        Response(),
        generator(),
    )
    count = 0
    had_done = False
    had_finish = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            assert had_finish
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            assert "id" in dct
            assert "object" in dct
            assert "choices" in dct
            if count == 1:
                assert dct["choices"][0]["delta"]["role"] == "assistant"
            elif dct["choices"][0]["finish_reason"]:
                assert dct["choices"][0]["finish_reason"] == "stop"
                had_finish = True
            else:
                assert dct["choices"][0]["delta"]["content"] == "abcd"
    assert had_done
    assert had_finish
    assert count == 14
    assert dct["usage"]["completion_tokens"] == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "generator", [fake_generator_with_error, fake_generator_with_error_batched]
)
async def test_chat_completions_stream_with_error(generator):
    generator = _chat_completions_wrapper(
        "1",
        VLLMChatCompletions(
            model="test", messages=[Message(role="user", content="test")]
        ),
        Request({"type": "http"}, {"request_id": "1"}),
        Response(),
        generator(),
    )
    count = 0
    had_error = False
    had_done = False
    async for x in generator:
        count += 1
        assert x.startswith("data: ")
        assert x.endswith("\n\n")
        if x.strip() == "data: [DONE]":
            had_done = True
        elif had_done:
            raise AssertionError()
        else:
            dct = json.loads(x[6:].strip())
            if "error" in dct:
                had_error = True
                assert dct["error"]["message"] == "error"
            elif had_error:
                raise AssertionError()
            else:
                assert "id" in dct
                assert "object" in dct
                assert "choices" in dct
                if count == 1:
                    assert dct["choices"][0]["delta"]["role"] == "assistant"
                elif not had_error:
                    assert dct["choices"][0]["delta"]["content"] == "abcd"
    assert dct["error"]
    assert had_error
    assert count == 8
