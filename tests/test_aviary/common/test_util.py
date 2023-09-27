import asyncio

import pytest

from aviary.backend.llm.vllm.util import BatchAviaryModelResponses
from aviary.backend.server.models import AviaryModelResponse


async def fake_generator():
    for _i in range(100):
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")


async def fake_generator_slow():
    for _i in range(100):
        await asyncio.sleep(0.01)
        yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")


class TestBatching:
    @pytest.mark.asyncio
    async def test_batch(self):
        count = 0
        batcher = BatchAviaryModelResponses(fake_generator())
        async for x in batcher.stream():
            count += 1
            assert x.num_generated_tokens == 100
            assert x.generated_text == "abcd" * 100

        # Should only have been called once
        assert count == 1
        assert batcher.queue.empty()

    @pytest.mark.asyncio
    async def test_batch_timing(self):
        count = 0
        batcher = BatchAviaryModelResponses(fake_generator_slow())
        async for _x in batcher.stream():
            count += 1

        assert (
            9 <= count <= 11
        ), "Count should have been called between 8 and 11 times, because each iteration takes 10ms to yield."
        assert batcher.queue.empty()

    @pytest.mark.asyncio
    async def test_exception_propagation(self):
        """Test that exceptions are propagated correctly to parent."""

        async def generator_should_raise():
            for _i in range(100):
                await asyncio.sleep(0.01)
                yield AviaryModelResponse(num_generated_tokens=1, generated_text="abcd")
                raise ValueError()

        count = 0
        batched = BatchAviaryModelResponses(generator_should_raise())

        async def parent():
            nonlocal count
            nonlocal batched
            async for _x in batched.stream():
                count += 1

        task = asyncio.create_task(parent())
        await asyncio.sleep(0.2)

        with pytest.raises(ValueError):
            task.result()
        assert count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("to_cancel", ["parent", "inner", "stream"])
    async def test_cancellation(self, to_cancel: str):
        """There are 3 ways cancellation can happen:
        1. The parent is cancelled
        2. The generator is cancelled
        3. The stream task is directly cancelled.

        Make sure all associated tasks are cancelled in each instance.
        """

        async def generator_should_raise():
            with pytest.raises(asyncio.CancelledError):
                for _i in range(100):
                    await asyncio.sleep(0.01)
                    yield AviaryModelResponse(
                        num_generated_tokens=1, generated_text="abcd"
                    )
                    if to_cancel == "inner":
                        raise asyncio.CancelledError()

        count = 0
        batched = BatchAviaryModelResponses(generator_should_raise())

        async def parent():
            nonlocal count
            nonlocal batched
            async for _x in batched.stream():
                count += 1

        task = asyncio.create_task(parent())
        await asyncio.sleep(0.2)

        cancel_task = {
            "parent": task,
            "stream": batched.read_task,
        }.get(to_cancel)

        if cancel_task:
            assert not task.done()
            assert not batched.read_task.done()
            cancel_task.cancel()

        await asyncio.sleep(0.3)
        assert batched.read_task.done(), "Read task should be completed"
        assert task.done(), "All tasks should be done"

        # Inner task is checked automatically with pytest.raises
