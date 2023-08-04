from abc import ABC, abstractmethod


class ValidationError(ValueError):
    error_code = 400
    pass


class PromptTooLongError(ValidationError):
    pass


class OutOfMemoryError(RuntimeError):
    error_code = 500
    pass


class ErrorReason(ABC):
    @abstractmethod
    def get_message(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_message()

    @abstractmethod
    def raise_exception(self):
        raise NotImplementedError


class InputTooLong(ErrorReason):
    def __init__(self, num_tokens: int, max_num_tokens: int) -> None:
        self.num_tokens = num_tokens
        self.max_num_tokens = max_num_tokens

    def get_message(self) -> str:
        return f"Input too long. Recieved {self.num_tokens} tokens, but the maximum input length is {self.max_num_tokens} tokens."

    def raise_exception(self):
        raise PromptTooLongError(self.get_message())


class OutOfMemory(ErrorReason):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def get_message(self) -> str:
        return self.msg

    def raise_exception(self):
        raise OutOfMemoryError(self.get_message())


class IrrecoverableError(ErrorReason):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def get_message(self) -> str:
        return self.msg

    def raise_exception(self):
        raise RuntimeError(self.get_message())
