import pytest
import ray

from rayllm.backend.server.models import AviaryModelResponse
from rayllm.backend.server.utils import (
    extract_message_from_exception,
    get_lines_batched,
    get_model_response_batched,
)


async def byte_generator(output):
    for x in output:
        assert isinstance(
            x, bytes
        ), f"Received ain input {x} which has type {type(x)}, instead of bytes"
        yield x


async def assert_generator_matches(generator, output):
    received = [x async for x in generator]
    assert received == output, f"\nReceived {received}\nexpected {output}"


async def assert_batched_generator_matches(generator, output):
    await assert_generator_matches(get_lines_batched(generator), output)


class TestStreaming:
    @pytest.mark.asyncio
    async def test_utils(self):
        # Bytes generator always returns bytes
        await assert_generator_matches(
            byte_generator([b"a", b"b", b"c"]), [b"a", b"b", b"c"]
        )

        # Assert generator actuall raises errors
        with pytest.raises(AssertionError):
            await assert_generator_matches(
                byte_generator([b"a", b"b"]), [b"a", b"b", b"c"]
            )

    @pytest.mark.asyncio
    async def test_collapses(self):
        output = [b"1", b"2", b"\n"]
        await assert_batched_generator_matches(byte_generator(output), [b"12\n"])

    @pytest.mark.asyncio
    async def test_multiline(self):
        output = [b"12\n34\n"]
        # No change
        await assert_batched_generator_matches(byte_generator(output), output)

    @pytest.mark.asyncio
    async def test_empty_lines(self):
        output = [b"\n\n\n\n"]
        await assert_batched_generator_matches(byte_generator(output), output)

    @pytest.mark.asyncio
    async def test_empty_lines_repeated(self):
        output = [b"\n\n\n\n"]
        output = [b"\n", b"\n", b"\n", b"\n"]
        await assert_batched_generator_matches(byte_generator(output), output)

    @pytest.mark.asyncio
    async def test_e2e(self):
        i = [
            b"partial,",
            b"partial,",
            b"done\ndone\ndone\n",
            b"done\n",
            b"partial,",
            b"partial,",
            b"done\n",
            b"partial,",
            b"\n",
        ]
        o = [
            b"partial,partial,done\ndone\ndone\n",
            b"done\n",
            b"partial,partial,done\n",
            b"partial,\n",
        ]
        await assert_batched_generator_matches(byte_generator(i), o)

    @pytest.mark.asyncio
    async def test_get_model_response_batched(self):
        output_str = "hi, this is your model speaking."

        # Simulate the response stream
        aviary_model_responses = [
            AviaryModelResponse(
                generated_text=c, num_input_tokens=1, num_generated_tokens=1
            )
            for c in output_str
        ]

        output_stream = "\n".join([a.json() for a in aviary_model_responses]).encode()
        l2 = len(output_stream) // 2

        # Output stream length should be roughly 15k
        assert l2 > 200, "Output stream is long"
        chunks = (
            output_stream[:100],
            output_stream[100:200],
            output_stream[200:l2],
            output_stream[l2:],
        )
        print("chunks", chunks)

        # Get a list of outputs
        output = [x async for x in get_model_response_batched(byte_generator(chunks))]

        merged = AviaryModelResponse.merge_stream(*aviary_model_responses)
        assert AviaryModelResponse.merge_stream(*output) == merged
        print([x.json() for x in output])

        assert len(output) == 3, "There should be 3 outputs produced"
        assert merged.generated_text == output_str


def test_extract_message_from_exception():
    e = ValueError("test")
    exc = extract_message_from_exception(e)
    assert exc == "ValueError: test"

    e = ValueError("test\n  test")
    exc = extract_message_from_exception(e)
    assert exc == "ValueError: test\n  test"

    @ray.remote
    def f():
        raise ValueError("test")

    try:
        ray.get(f.remote())
    except ValueError as e:
        exc = extract_message_from_exception(e)
        assert exc == "ValueError: test"

    @ray.remote
    def f():
        raise ValueError("test\n  test")

    try:
        ray.get(f.remote())
    except ValueError as e:
        exc = extract_message_from_exception(e)
        assert exc == "ValueError: test\n  test"
