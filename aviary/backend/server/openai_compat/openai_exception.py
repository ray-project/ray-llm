class OpenAIHTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        type: str = "Unknown",
    ) -> None:
        self.status_code = status_code
        self.message = message
        self.type = type
