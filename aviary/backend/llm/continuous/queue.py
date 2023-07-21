import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Optional, Union

import ray

from aviary.backend.llm.continuous.tokenstream import TokenStream

from .error_handling import ErrorReason
from .types import Request


@dataclass
class InferenceRequest:
    id: int
    request: Request
    output_stream: TokenStream
    submit_time_ns: int
    request_input_length: Union[int, ray.ObjectRef]

    def mark_as_invalid(self, reason: ErrorReason):
        self.output_stream.put(reason)
        self.mark_as_finished()

    def mark_as_finished(self):
        self.output_stream.end()

    @property
    def is_finished(self):
        return self.output_stream.is_finished

    @property
    def request_input_length(self) -> int:
        self._request_input_length = (
            ray.get(self._request_input_length)
            if isinstance(self._request_input_length, ray.ObjectRef)
            else self._request_input_length
        )
        self.output_stream.num_input_tokens = self._request_input_length
        return self._request_input_length

    @request_input_length.setter
    def request_input_length(self, v: Union[int, ray.ObjectRef]) -> None:
        self._request_input_length = v

    @classmethod
    def from_request(cls, request: Request, request_input_length: int):
        return cls(
            id=request.id,
            request=request,
            request_input_length=request_input_length,
            output_stream=TokenStream(request.id),
            submit_time_ns=int(time.time()),
        )

    @property
    def total_tokens(self) -> int:
        return self.request_input_length + self.request.max_new_tokens

    @property
    def input_length(self) -> int:
        return self.request_input_length + self.output_stream.num_tokens()

    @property
    def gen_length(self) -> int:
        return max(
            0,
            self.request.max_new_tokens - self.output_stream.num_tokens(),
        )


class RequestQueue:
    def __init__(self):
        self._queue = deque()
        self._lock = RLock()
        self._cv = Condition(self._lock)

    def push(self, request: InferenceRequest) -> bool:
        with self._cv:
            self._queue.append(request)
            self._cv.notify_all()
            return True

    def peek(self) -> Optional[InferenceRequest]:
        with self._lock:
            if len(self._queue) == 0:
                return None
            return self._queue[0]

    def pop(self) -> Optional[InferenceRequest]:
        with self._lock:
            while len(self._queue) == 0:
                return None
            return self._queue.popleft()

    def wait(self, timeout=None):
        start = time.time()
        with self._cv:
            while len(self._queue) == 0:
                self._cv.wait(timeout)
                if timeout is not None and time.time() - start >= timeout:
                    return

    def reverse_push(self, request: InferenceRequest) -> None:
        with self._cv:
            self._queue.appendleft(request)
            self._cv.notify_all()

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
