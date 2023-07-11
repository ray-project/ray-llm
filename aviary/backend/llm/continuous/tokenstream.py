import asyncio


class TokenStream:
    """A stream of tokens that can be iterated over asynchronously."""

    def __init__(self, id: int):
        self.id = id
        self.num_input_tokens = None
        self._queue = asyncio.Queue()
        self._num_tokens = 0
        self._generated_text = None
        self.is_finished = False

    def end(self, generated_text=None):
        self.is_finished = True
        self._generated_text = generated_text
        self._queue.put_nowait(StopIteration)

    def put(self, item):
        if not self.is_finished:
            self._queue.put_nowait(item)
            self._num_tokens += 1

    def num_tokens(self):
        return self._num_tokens

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if result == StopIteration:
            raise StopAsyncIteration
        return result
