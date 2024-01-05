from typing import Union

from fastapi import HTTPException, Request
from httpx import HTTPStatusError as HTTPXHTTPStatusError
from opentelemetry import trace
from starlette.responses import JSONResponse

from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.utils import get_response_for_error


def openai_exception_handler(
    request: Request,
    exc: Union[OpenAIHTTPException, HTTPException],
):
    assert isinstance(
        exc, (OpenAIHTTPException, HTTPException, HTTPXHTTPStatusError)
    ), f"Unable to handle invalid exception {type(exc)}"

    err_response = get_response_for_error(
        exc, request.state.request_id, prefix="Returning error to user"
    )

    span = trace.get_current_span()
    span.record_exception(exc)
    span.set_status(trace.StatusCode.ERROR, description=err_response.error.message)

    return JSONResponse(content=err_response.dict(), status_code=exc.status_code)
