import traceback

from fastapi import Request
from starlette.responses import JSONResponse

from aviary.backend.server.models import AviaryModelResponse
from aviary.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from aviary.common.models import ErrorResponse


def openai_exception_handler(request: Request, exc: OpenAIHTTPException):
    assert isinstance(
        exc, OpenAIHTTPException
    ), f"Unable to handle invalid exception {type(exc)}"
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    err_response = AviaryModelResponse(
        error=ErrorResponse(
            message=exc.message,
            code=exc.status_code,
            internal_message=message,
            type=exc.type,
        )
    )
    return JSONResponse(content=err_response.dict(), status_code=exc.status_code)
