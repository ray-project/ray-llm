import asyncio
import os
import traceback
from functools import partial
from typing import (
    AsyncIterable,
    Awaitable,
    Callable,
    List,
    Optional,
    TypeVar,
    Union,
)

import aiohttp
import pydantic
from fastapi import HTTPException, Request, status
from httpx import HTTPStatusError as HTTPXHTTPStatusError
from opentelemetry import trace
from pydantic import ValidationError as PydanticValidationError

from rayllm.backend.logger import get_logger
from rayllm.backend.server import constants
from rayllm.backend.server.models import (
    AviaryModelResponse,
    LLMApp,
)
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.common.models import ErrorResponse

T = TypeVar("T")

logger = get_logger(__name__)
AVIARY_ROUTER_HTTP_TIMEOUT = float(os.environ.get("AVIARY_ROUTER_HTTP_TIMEOUT", 175))


def parse_args(
    args: Union[str, LLMApp, List[Union[LLMApp, str]]], llm_app_cls=LLMApp
) -> List[LLMApp]:
    """Parse the input args and return a standardized list of LLMApp objects

    Supported args format:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    """

    raw_models = []
    if isinstance(args, list):
        raw_models = args
    else:
        raw_models = [args]

    # For each
    models: List[LLMApp] = []
    for raw_model in raw_models:
        if isinstance(raw_model, str):
            if os.path.exists(raw_model):
                parsed_models = _parse_path_args(raw_model, llm_app_cls=llm_app_cls)
            else:
                try:
                    parsed_models = [llm_app_cls.parse_yaml(raw_model)]
                except pydantic.ValidationError as e:
                    if "__root__" in repr(e):
                        raise ValueError(
                            "Could not parse string as yaml. If you are specifying a path, make sure it exists and can be reached."
                        ) from e
                    else:
                        raise
        else:
            parsed_models = [llm_app_cls.parse_obj(raw_model)]
        models += parsed_models
    return [model for model in models if model.enabled]


def _parse_path_args(path: str, llm_app_cls=LLMApp) -> List[LLMApp]:
    assert os.path.exists(
        path
    ), f"Could not load model from {path}, as it does not exist."
    if os.path.isfile(path):
        with open(path, "r") as f:
            return [llm_app_cls.parse_yaml(f)]
    elif os.path.isdir(path):
        apps = []
        for root, _dirs, files in os.walk(path):
            for p in files:
                if _is_yaml_file(p):
                    with open(os.path.join(root, p), "r") as f:
                        apps.append(llm_app_cls.parse_yaml(f))
        return apps
    else:
        raise ValueError(
            f"Could not load model from {path}, as it is not a file or directory."
        )


def _is_yaml_file(filename: str) -> bool:
    yaml_exts = [".yml", ".yaml", ".json"]
    for s in yaml_exts:
        if filename.endswith(s):
            return True
    return False


def _replace_prefix(model: str) -> str:
    """Replace -- with / in model name to handle slashes within the URL path segment"""
    return model.replace("--", "/")


async def _until_disconnected(request: Request):
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(1)


EOS_SENTINELS = (None, StopIteration, StopAsyncIteration)


async def collapse_stream(async_iterator: AsyncIterable[T]) -> List[T]:
    return [x async for x in async_iterator]


async def get_lines_batched(async_iterator: AsyncIterable[bytes]):
    """Batch the lines of a bytes iterator

    Group the output of an async iterator into lines.

    Eg. if the iterator output:
    b'a'
    b'b'
    b'c\n'

    The output would be
    b'abc\n'

    Furthermore, if multiple lines are present, return all of them.

    Args:
        async_iterator (AsyncIterable[bytes]): A bytes iterator

    Yields:
        AsyncIterable[bytes]: A bytes iterator that chunks on lines
    """
    remainder = b""
    async for chunk in async_iterator:
        remainder += chunk
        if b"\n" in remainder:
            out, remainder = remainder.rsplit(b"\n", 1)
            yield out + b"\n"

    if remainder != b"":
        yield remainder


async def get_model_response_batched(async_iterator: AsyncIterable[bytes]):
    """Parse AviaryModelResponse from byte stream

    First group the data from the iterator into lines.
    Then for each line, parse it as an AviaryModelResponse object.

    Args:
        async_iterator (AsyncIterable[bytes]): the input iterator

    Yields:
        AviaryModelResponse
    """
    async for chunk in get_lines_batched(async_iterator):
        responses = [AviaryModelResponse.parse_raw(p) for p in chunk.split(b"\n") if p]
        combined_response = AviaryModelResponse.merge_stream(*responses)
        yield combined_response


async def stream_model_responses(
    url: str, json=None, timeout=AVIARY_ROUTER_HTTP_TIMEOUT
):
    """Make a streaming network request, and parse the output into a stream of AviaryModelResponse

    Take the output stream of the request and parse it into a stream of AviaryModelResponse.

    Args:
        url (str): The url to querky
        json (_type_, optional): the json body
        timeout (_type_, optional): Defaults to AVIARY_ROUTER_HTTP_TIMEOUT.

    Yields:
        AviaryModelResponse
    """
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(
            url,
            json=json,
            timeout=timeout,
        ) as response:
            async for combined_response in get_model_response_batched(
                response.content.iter_any()
            ):
                yield combined_response


T = TypeVar("T")


def make_async(_func: Callable[..., T]) -> Callable[..., Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args, **kwargs) -> asyncio.Future:
        loop = asyncio.get_event_loop()
        func = partial(_func, *args, **kwargs)
        return loop.run_in_executor(executor=None, func=func)

    return _async_wrapper


def extract_message_from_exception(e: Exception) -> str:
    # If the exception is a Ray exception, we need to dig through the text to get just
    # the exception message without the stack trace
    # This also works for normal exceptions (we will just return everything from
    # format_exception_only in that case)
    message_lines = traceback.format_exception_only(type(e), e)[-1].strip().split("\n")
    message = ""
    # The stack trace lines will be prefixed with spaces, so we need to start from the bottom
    # and stop at the last line before a line with a space
    found_last_line_before_stack_trace = False
    for line in reversed(message_lines):
        if not line.startswith(" "):
            found_last_line_before_stack_trace = True
        if found_last_line_before_stack_trace and line.startswith(" "):
            break
        message = line + "\n" + message
    message = message.strip()
    return message


def get_response_for_error(
    e: Exception,
    request_id: str,
    span: Optional[trace.Span] = None,
    prefix="",
    enable_returning_500_errors: Optional[bool] = None,
) -> AviaryModelResponse:
    """Convert an exception to an AviaryModelResponse object"""
    enable_returning_500_errors = (
        enable_returning_500_errors
        if enable_returning_500_errors is not None
        else constants.enable_returning_internal_exceptions
    )
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(e, HTTPException):
        status_code = e.status_code
    elif isinstance(e, OpenAIHTTPException):
        status_code = e.status_code
    elif isinstance(e, PydanticValidationError):
        status_code = 400
    elif isinstance(e, HTTPXHTTPStatusError):
        status_code = e.response.status_code
    else:
        # Try to get the status code attribute
        status_code = getattr(e, "status_code", status_code)

    logger.error(
        f"{prefix}. Request {request_id} failed with status code {status_code}: {e}",
        exc_info=e,
    )

    if (
        status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        and not enable_returning_500_errors
    ):
        message = "Internal Server Error"
        internal_message = message
        exc_type = "InternalServerError"
    else:
        if isinstance(e, OpenAIHTTPException) and e.internal_message is not None:
            internal_message = e.internal_message
        else:
            internal_message = extract_message_from_exception(e)
        if isinstance(e, HTTPException):
            message = e.detail
        elif isinstance(e, OpenAIHTTPException):
            message = e.message
        else:
            message = internal_message
        exc_type = e.__class__.__name__

    # TODO make this more robust
    if "(Request ID: " not in message:
        message += f" (Request ID: {request_id})"

    if "(Request ID: " not in internal_message:
        internal_message += f" (Request ID: {request_id})"

    if span is None:
        span = trace.get_current_span()
    span.record_exception(e)
    span.set_status(trace.StatusCode.ERROR, description=message)

    return AviaryModelResponse(
        error=ErrorResponse(
            message=message,
            code=status_code,
            internal_message=internal_message,
            type=exc_type,
        ),
    )
