import asyncio
from typing import AsyncIterator

from rayllm.backend.server.models import (
    AviaryModelResponse,
    BatchedAviaryModelResponse,
)


class BatchAviaryModelResponses:
    """This class batches AviaryModelResponses from a generator into a single response, at some time interval."""

    def __init__(self, generator: AsyncIterator[AviaryModelResponse], interval_ms=100):
        self.generator = generator
        self.queue: asyncio.Queue = asyncio.Queue()
        self.interval_s = interval_ms / 1000

        # We are okay with this task getting cancelled (to propogate cancellations)
        self.read_task = asyncio.create_task(self.read())

    async def stream(self) -> AsyncIterator[BatchedAviaryModelResponse]:
        """Drain from the queue every interval_ms and yield the merged results"""
        try:
            while True:
                # Wait for the interval
                await asyncio.sleep(self.interval_s)

                # Get all elements from the queue
                results, is_done = self.check_done_and_drain()

                # If there are results, merge and yield them
                if results:
                    output: BatchedAviaryModelResponse = BatchedAviaryModelResponse.merge_stream(*results)  # type: ignore
                    yield output

                # If the read task is done, exit the stream task
                if is_done:
                    # Raise exception, if any
                    self.read_task.result()
                    break
        finally:
            # If the stream task is done, make sure to exit the read task
            if not self.read_task.done():
                self.read_task.cancel()

    def check_done_and_drain(self):
        results = self.drain_queue()
        return results, self.read_task.done()

    async def read(self):
        """Read from the generator and put into the queue in a tight loop"""
        async for x in self.generator:
            self.queue.put_nowait(x)

    def drain_queue(self):
        """Drain all results currently in the queue"""
        results = []
        try:
            while True:
                results.append(self.queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return results
