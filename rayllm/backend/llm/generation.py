import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from rayllm.backend.llm.error_handling import ErrorReason


class FinishReason(str, Enum):
    LENGTH = "length"
    STOP = "stop"
    ERROR = "error"
    CANCELLED = "cancelled"
    TOOL_CALLS = "tool_calls"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_vllm_finish_reason(
        cls, finish_reason: Optional[str]
    ) -> Optional["FinishReason"]:
        if finish_reason is None:
            return None
        if finish_reason == "stop":
            return cls.STOP
        if finish_reason == "length":
            return cls.LENGTH
        if finish_reason == "abort":
            return cls.CANCELLED
        return cls.STOP


@dataclass
class GenerationMetadata:
    num_input_tokens: int
    num_output_tokens: int
    generation_time_s: float


@dataclass
class Generation:
    text: str
    metadata: GenerationMetadata


class GenerationStream:
    """A stream of tokens elements (tokens, embeddings, etc) can be iterated over asynchronously."""

    def __init__(self, id: int):
        self.id = id
        self._queue: asyncio.Queue = asyncio.Queue()
        self._num_generations = 0
        self._submit_time_s = time.time()
        self.first_generation_time_s: Optional[float] = None
        self.finish_time_s: Optional[float] = None
        self.last_generation_time_s: Optional[float] = None
        self.finish_reason: Optional[FinishReason] = None
        self.error_reason: Optional[ErrorReason] = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None

    @property
    def num_generations(self) -> int:
        return self._num_generations

    @property
    def submit_time_s(self) -> float:
        return self._submit_time_s

    @property
    def time_to_first_generation_s(self) -> Optional[float]:
        if self.first_generation_time_s is None:
            return None
        return self.first_generation_time_s - self.submit_time_s

    @property
    def total_time_s(self) -> Optional[float]:
        if self.finish_time_s is None:
            return None
        return self.finish_time_s - self.submit_time_s

    @property
    def avg_generation_time_s(self) -> Optional[float]:
        if (
            self.num_generations == 0
            or self.last_generation_time_s is None
            or self.first_generation_time_s is None
        ):
            return None
        return (
            self.last_generation_time_s - self.first_generation_time_s
        ) / self.num_generations

    def end(
        self,
        finish_reason: FinishReason,
        error_reason: Optional[ErrorReason] = None,
    ):
        if self.is_finished:
            return
        self.finish_reason = finish_reason
        self.error_reason = error_reason
        self._queue.put_nowait(StopIteration)

    def put(self, item: Generation):
        if self.is_finished:
            return
        if self.first_generation_time_s is None:
            self.first_generation_time_s = time.time()
        self.last_generation_time_s = time.time()
        self._queue.put_nowait(item)
        self._num_generations += 1

    def __aiter__(self):
        return self

    async def __anext__(self) -> Generation:
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        return result
