import asyncio
import json
import os
from enum import IntEnum
from typing import AsyncIterable, AsyncIterator, List, TypeVar, Union

import pydantic
from fastapi import Request
from starlette.responses import StreamingResponse

from aviary.backend.server.models import LLMApp

T = TypeVar("T")


def parse_args(args: Union[str, LLMApp, List[Union[LLMApp, str]]]) -> List[LLMApp]:
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
                parsed_models = _parse_path_args(raw_model)
            else:
                try:
                    parsed_models = [LLMApp.parse_yaml(raw_model)]
                except pydantic.ValidationError as e:
                    if "__root__" in repr(e):
                        raise ValueError(
                            "Could not parse string as yaml. If you are specifying a path, make sure it exists and can be reached."
                        ) from e
                    else:
                        raise
        else:
            parsed_models = [LLMApp.parse_obj(raw_model)]
        models += parsed_models
    return [model for model in models if model.enabled]


def _parse_path_args(path: str) -> List[LLMApp]:
    assert os.path.exists(
        path
    ), f"Could not load model from {path}, as it does not exist."
    if os.path.isfile(path):
        with open(path, "r") as f:
            return [LLMApp.parse_yaml(f)]
    elif os.path.isdir(path):
        apps = []
        for root, _dirs, files in os.walk(path):
            for p in files:
                if _is_yaml_file(p):
                    with open(os.path.join(root, p), "r") as f:
                        apps.append(LLMApp.parse_yaml(f))
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
    return model.replace("--", "/")


async def _until_disconnected(request: Request):
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(1)


EOS_SENTINELS = (None, StopIteration, StopAsyncIteration)


async def serialize_stream(async_iterator: AsyncIterator):
    try:
        async for x in async_iterator:
            if isinstance(x, pydantic.BaseModel):
                x = x.json() + "\n"

            if not isinstance(x, (str, bytes)):
                raise TypeError(f"Unable to serialize object {x}, of type {type(x)}")

            yield x
    except Exception as e:
        err = {"error": f"Internal server error: {e}"}
        yield json.dumps(err) + "\n"


def get_streaming_response(async_iterator: AsyncIterable) -> StreamingResponse:
    return StreamingResponse(
        serialize_stream(async_iterator),
        media_type="text/event-stream",
    )


async def collapse_stream(async_iterator: AsyncIterable[T]) -> List[T]:
    return [x async for x in async_iterator]


class QueuePriority(IntEnum):
    """Lower value = higher priority"""

    GENERATE_TEXT = 0
    BATCH_GENERATE_TEXT = 1
