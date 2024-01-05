from abc import ABC, abstractmethod, abstractproperty


class ValidationError(ValueError):
    status_code = 400
    pass


class PromptTooLongError(ValidationError):
    pass


class TooManyStoppingSequencesError(ValidationError):
    pass


class OutOfMemoryError(RuntimeError):
    status_code = 500
    pass


class ErrorReason(ABC):
    @abstractmethod
    def get_message(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_message()

    @abstractproperty
    def exception(self) -> Exception:
        raise NotImplementedError

    def raise_exception(self) -> Exception:
        raise self.exception


class InputTooLong(ErrorReason):
    def __init__(self, num_tokens: int, max_num_tokens: int) -> None:
        self.num_tokens = num_tokens
        self.max_num_tokens = max_num_tokens

    def get_message(self) -> str:
        if self.num_tokens < 0:
            return f"Input too long. The maximum input length is {self.max_num_tokens} tokens."
        return f"Input too long. Recieved {self.num_tokens} tokens, but the maximum input length is {self.max_num_tokens} tokens."

    @property
    def exception(self) -> Exception:
        return PromptTooLongError(self.get_message())


class TooManyStoppingSequences(ErrorReason):
    def __init__(
        self, num_stopping_sequences: int, max_num_stopping_sequences: int
    ) -> None:
        self.num_stopping_sequences = num_stopping_sequences
        self.max_num_stopping_sequences = max_num_stopping_sequences

    def get_message(self) -> str:
        return (
            f"Too many stopping sequences. Recieved {self.num_stopping_sequences} stopping sequences,"
            f"but the maximum is {self.max_num_stopping_sequences}. Please reduce the number of provided stopping sequences."
        )

    @property
    def exception(self) -> Exception:
        return TooManyStoppingSequencesError(self.get_message())


class OutOfMemory(ErrorReason):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def get_message(self) -> str:
        return self.msg

    @property
    def exception(self) -> Exception:
        return OutOfMemoryError(self.get_message())


class IrrecoverableError(ErrorReason):
    def __init__(self, msg: str) -> None:
        self.msg = msg

    def get_message(self) -> str:
        return self.msg

    @property
    def exception(self) -> Exception:
        return RuntimeError(self.get_message())
