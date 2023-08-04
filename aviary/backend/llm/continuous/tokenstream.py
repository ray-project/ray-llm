import asyncio
from enum import Enum
from typing import Optional

from .error_handling import ErrorReason


class FinishReason(str, Enum):
    LENGTH = "length"
    STOP = "stop"
    ERROR = "error"
    CANCELLED = "cancelled"

    @classmethod
    def from_tgi_finish_reason(cls, finish_reason: int) -> "FinishReason":
        if finish_reason == 0:
            return cls.LENGTH
        return cls.STOP

    def __str__(self) -> str:
        return self.value


class TokenStream:
    """A stream of tokens that can be iterated over asynchronously."""

    def __init__(self, id: int):
        self.id = id
        self._queue = asyncio.Queue()
        self._num_tokens = 0
        self._generated_text = None
        self.finish_reason: Optional[FinishReason] = None
        self.error_reason: Optional[ErrorReason] = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None

    def end(
        self,
        finish_reason: FinishReason,
        error_reason: Optional[ErrorReason] = None,
        generated_text: Optional[str] = None,
    ):
        if self.is_finished:
            return
        self.finish_reason = finish_reason
        self.error_reason = error_reason
        self._generated_text = generated_text
        self._queue.put_nowait(StopIteration)

    def put(self, item: str):
        if self.is_finished:
            return
        if self._num_tokens == 0:
            item = item.lstrip()
        self._queue.put_nowait(item)
        self._num_tokens += 1

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        return result
