from typing import Optional

from rayllm.backend.server.models import AviaryModelResponse


class OpenAIHTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        type: str = "Unknown",
        internal_message: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.type = type
        self.internal_message = internal_message

    @classmethod
    def from_model_response(
        cls, response: AviaryModelResponse
    ) -> "OpenAIHTTPException":
        return cls(
            status_code=response.error.code,
            message=response.error.message,
            type=response.error.type,
            internal_message=response.error.internal_message,
        )
