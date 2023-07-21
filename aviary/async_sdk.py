import json
from collections import namedtuple
from typing import AsyncIterator, Dict, Union

import aiohttp

from aviary.common.constants import DEFAULT_API_VERSION, TIMEOUT
from aviary.common.utils import (
    ResponseError,
    _is_aviary_model,
)

from .sdk import get_aviary_backend

__all__ = ["stream"]

response = namedtuple("Response", ["text", "status_code"])


# TODO: Add other methods


async def stream(
    model: str, prompt: str, version: str = DEFAULT_API_VERSION
) -> AsyncIterator[Dict[str, Union[str, float, int]]]:
    """Query Aviary and stream response"""
    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + model.replace("/", "--") + "/" + version + "stream"

        chunk = b""
        try:
            async with aiohttp.ClientSession(
                raise_for_status=True, headers=backend.header
            ) as session:
                async with session.post(
                    url, json={"prompt": prompt}, timeout=TIMEOUT[-1]
                ) as r:
                    async for chunk in r.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        data = json.loads(chunk)
                        if data.get("error"):
                            raise ResponseError(
                                data["error"],
                                response=response(data["error"], r.status),
                            )
                        yield data
        except Exception as e:
            if isinstance(e, ResponseError):
                raise e
            else:
                raise ResponseError(
                    str(e), response=response(chunk.decode("utf-8"), r.status)
                ) from e
    else:
        # TODO implement streaming for langchain models
        raise RuntimeError("Streaming is currently only supported for aviary models")
