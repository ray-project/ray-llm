import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Optional

from ray.serve.batching import (
    _BatchQueue,
    _SingleRequest,
)


class QueuePriority(IntEnum):
    """Lower value = higher priority"""

    GENERATE_TEXT = 0
    BATCH_GENERATE_TEXT = 1


@dataclass(order=True)
class _PriorityWrapper:
    """Wrapper allowing for priority queueing of arbitrary objects."""

    obj: Any = field(compare=False)
    priority: int = field(compare=True)


class PriorityQueueWithUnwrap(asyncio.PriorityQueue):
    def get_nowait(self) -> Any:
        # Get just the obj from _PriorityWrapper
        ret: _PriorityWrapper = super().get_nowait()
        return ret.obj


class _PriorityBatchQueue(_BatchQueue):
    # The kwarg of the batch function used to determine priority.
    _priority_kwarg: str = "priority"

    def __init__(
        self,
        max_batch_size: int,
        timeout_s: float,
        handle_batch_func: Optional[Callable] = None,
    ) -> None:
        """Async queue that accepts individual items and returns batches.

        Compared to base _BatchQueue, this class uses asyncio.PriorityQueue.

        Respects max_batch_size and timeout_s; a batch will be returned when
        max_batch_size elements are available or the timeout has passed since
        the previous get.

        If handle_batch_func is passed in, a background coroutine will run to
        poll from the queue and call handle_batch_func on the results.

        Arguments:
            max_batch_size: max number of elements to return in a batch.
            timeout_s: time to wait before returning an incomplete
                batch.
            handle_batch_func(Optional[Callable]): callback to run in the
                background to handle batches if provided.
        """
        super().__init__(max_batch_size, timeout_s, handle_batch_func)
        self.queue: PriorityQueueWithUnwrap[_SingleRequest] = PriorityQueueWithUnwrap()

    def put(
        self,
        request: _SingleRequest,
    ) -> None:
        # Lower index = higher priority
        priority = int(
            request.flattened_args[
                request.flattened_args.index(self._priority_kwarg) + 1
            ]
        )
        super().put(_PriorityWrapper(obj=request, priority=int(priority)))
