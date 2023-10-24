import asyncio
import gc
import time

import pytest

from rayllm.backend.server.plugins.execution_hooks import ShieldedTaskSet
from rayllm.backend.server.utils import make_async


class AsyncCounter:
    def __init__(self):
        self.count = 0

    async def incr(self):
        await asyncio.sleep(0.2)  # sleep for 200 ms before incrementing
        self.count += 1

    def reset(self):
        self.count = 0


@pytest.mark.asyncio
async def test_task_set():
    task_set = ShieldedTaskSet()

    counter = AsyncCounter()

    await task_set.run(counter.incr())
    assert (
        counter.count == 1
    ), "Running the shielded command should increment the counter"

    counter.reset()

    async def _to_cancel():
        try:
            # This coroutine should be cancelled before completing
            with pytest.raises(asyncio.CancelledError):
                await asyncio.sleep(100)
        finally:
            task_set.run(counter.incr())

    async def _to_cancel_parallel(x):
        await asyncio.gather(*(_to_cancel() for _ in range(x)))

    p = 10
    cancellable = asyncio.create_task(_to_cancel_parallel(p))

    # Wait 1 second and cancel the top level coroutine
    await asyncio.sleep(1)
    cancellable.cancel()
    gc.collect()

    # Wait 1 second for all coroutines to be scheduled
    await asyncio.sleep(1)

    assert (
        counter.count == p
    ), f"All {p} coroutines should have run the shielded coroutine"


@pytest.mark.asyncio
async def test_task_set_generator():
    task_set = ShieldedTaskSet()

    counter = AsyncCounter()
    assert counter.count == 0

    async def _to_cancel():
        try:
            # This coroutine should be cancelled before completing
            with pytest.raises(asyncio.CancelledError):
                for i in range(100):
                    yield i
                    await asyncio.sleep(0.1)
        finally:
            task_set.run(counter.incr())

    stream = _to_cancel()
    async for x in stream:
        if x > 3:
            await stream.aclose()
            gc.collect()

    # Wait 1 second and cancel the top level coroutine
    await asyncio.sleep(1)

    assert counter.count == 1, "The generator should have run the counter increment"


@pytest.mark.asyncio
async def test_make_async():
    @make_async
    def sync_blocking():
        time.sleep(1)

    async_counter = AsyncCounter()

    async def incr_in_loop():
        for _i in range(10):
            await async_counter.incr()

    task = asyncio.create_task(incr_in_loop())
    await sync_blocking()

    assert (
        async_counter.count >= 4
    ), "The async counter should have been incremented 4 times while the time.sleep was running."
    task.cancel()
