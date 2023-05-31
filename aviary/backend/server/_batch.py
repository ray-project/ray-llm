import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple, Type

from ray.serve.batching import (
    _BatchQueue,
    _SingleRequest,
    extract_signature,
    flatten_args,
    get_or_create_event_loop,
    iscoroutinefunction,
)

# TODO: Upstream to Serve.


def extract_self_if_method_call(args: List[Any], func: Callable) -> Optional[object]:
    """Check if this is a method rather than a function.

    Does this by checking to see if `func` is the attribute of the first
    (`self`) argument under `func.__name__`. Unfortunately, this is the most
    robust solution to this I was able to find. It would also be preferable
    to do this check when the decorator runs, rather than when the method is.

    Returns the `self` object if it's a method call, else None.

    Arguments:
        args: arguments to the function/method call.
        func: the unbound function that was called.
    """
    if len(args) > 0:
        method = getattr(args[0], func.__name__, False)
        if method:
            wrapped = getattr(method, "__wrapped__", False)
            if wrapped and wrapped == func:
                return args[0]

    return None


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
        request: Tuple[_SingleRequest, asyncio.Future],
        *,
        priority: int,
    ) -> None:
        # Lower index = higher priority
        super().put(_PriorityWrapper(obj=request, priority=int(priority)))


def _validate_max_batch_size(max_batch_size):
    if not isinstance(max_batch_size, int):
        if isinstance(max_batch_size, float) and max_batch_size.is_integer():
            max_batch_size = int(max_batch_size)
        else:
            raise TypeError("max_batch_size must be integer >= 1")

    if max_batch_size < 1:
        raise ValueError("max_batch_size must be an integer >= 1")


def _validate_batch_wait_timeout_s(batch_wait_timeout_s):
    if not isinstance(batch_wait_timeout_s, (float, int)):
        raise TypeError("batch_wait_timeout_s must be a float >= 0")

    if batch_wait_timeout_s < 0:
        raise ValueError("batch_wait_timeout_s must be a float >= 0")


def batch(
    _func: Optional[Callable] = None,
    max_batch_size: int = 10,
    batch_wait_timeout_s: float = 0.0,
    *,
    batch_queue_cls: Type[_BatchQueue] = _BatchQueue,
):
    """Converts a function to asynchronously handle batches.

    The function can be a standalone function or a class method. In both
    cases, the function must be `async def` and take a list of objects as
    its sole argument and return a list of the same length as a result.

    When invoked, the caller passes a single object. These will be batched
    and executed asynchronously once there is a batch of `max_batch_size`
    or `batch_wait_timeout_s` has elapsed, whichever occurs first.

    Example:

    .. code-block:: python

            from ray import serve
            from starlette.requests import Request

            @serve.deployment
            class BatchedDeployment:
                @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
                async def batch_handler(self, requests: List[Request]) -> List[str]:
                    response_batch = []
                    for r in requests:
                        name = (await requests.json())["name"]
                        response_batch.append(f"Hello {name}!")

                    return response_batch

                async def __call__(self, request: Request):
                    return await self.batch_handler(request)

            app = BatchedDeployment.bind()

    Arguments:
        max_batch_size: the maximum batch size that will be executed in
            one call to the underlying function.
        batch_wait_timeout_s: the maximum duration to wait for
            `max_batch_size` elements before running the current batch.
        batch_queue_cls: the class to use for the batch queue.
    """
    # `_func` will be None in the case when the decorator is parametrized.
    # See the comment at the end of this function for a detailed explanation.
    if _func is not None:
        if not callable(_func):
            raise TypeError(
                "@serve.batch can only be used to decorate functions or methods."
            )

        if not iscoroutinefunction(_func):
            raise TypeError("Functions decorated with @serve.batch must be 'async def'")

    if not callable(max_batch_size):
        _validate_max_batch_size(max_batch_size)

    if not callable(batch_wait_timeout_s):
        _validate_batch_wait_timeout_s(batch_wait_timeout_s)

    def _batch_decorator(_func):
        @wraps(_func)
        async def batch_wrapper(*args, **kwargs):
            priority_kwarg = getattr(batch_queue_cls, "_priority_kwarg", None)
            priority_kwargs = {}
            if priority_kwarg:
                priority_kwargs = {priority_kwarg: kwargs.pop(priority_kwarg)}
            self = extract_self_if_method_call(args, _func)
            flattened_args: List = flatten_args(extract_signature(_func), args, kwargs)

            if self is None:
                # For functions, inject the batch queue as an
                # attribute of the function.
                batch_queue_object = _func
            else:
                # For methods, inject the batch queue as an
                # attribute of the object.
                batch_queue_object = self
                # Trim the self argument from methods
                flattened_args = flattened_args[2:]

            # The first time the function runs, we lazily construct the batch
            # queue and inject it under a custom attribute name. On subsequent
            # runs, we just get a reference to the attribute.
            batch_queue_attr = f"__serve_batch_queue_{_func.__name__}"
            if not hasattr(batch_queue_object, batch_queue_attr):
                batch_queue = batch_queue_cls(
                    max_batch_size, batch_wait_timeout_s, _func
                )
                setattr(batch_queue_object, batch_queue_attr, batch_queue)
            else:
                batch_queue = getattr(batch_queue_object, batch_queue_attr)

            if callable(max_batch_size):
                new_max_batch_size = max_batch_size(batch_queue_object)
                _validate_max_batch_size(new_max_batch_size)
                batch_queue.max_batch_size = new_max_batch_size

            if callable(batch_wait_timeout_s):
                new_batch_wait_timeout_s = batch_wait_timeout_s(batch_queue_object)
                _validate_batch_wait_timeout_s(new_batch_wait_timeout_s)
                batch_queue.timeout_s = new_batch_wait_timeout_s

            future = get_or_create_event_loop().create_future()
            batch_queue.put(
                _SingleRequest(self, flattened_args, future), **priority_kwargs
            )

            # This will raise if the underlying call raised an exception.
            return await future

        return batch_wrapper

    # Unfortunately, this is required to handle both non-parametrized
    # (@serve.batch) and parametrized (@serve.batch(**kwargs)) usage.
    # In the former case, `serve.batch` will be called with the underlying
    # function as the sole argument. In the latter case, it will first be
    # called with **kwargs, then the result of that call will be called
    # with the underlying function as the sole argument (i.e., it must be a
    # "decorator factory.").
    return _batch_decorator(_func) if callable(_func) else _batch_decorator
