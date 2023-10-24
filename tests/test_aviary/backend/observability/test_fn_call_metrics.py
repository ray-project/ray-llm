import asyncio
from typing import List
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rayllm.backend.observability.fn_call_metrics import TracingAsyncIterator


async def arange(stop):
    for i in range(stop):
        yield i


@pytest.fixture()
def local_tracing_context():
    tracer_provider = TracerProvider()

    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = tracer_provider.get_tracer(__name__)

    yield tracer, exporter


def run_aiter(tracer, async_iterator):
    async def test():
        with TracingAsyncIterator("test", async_iterator) as iterator:
            async for i in iterator:
                print(i)

    with patch("rayllm.backend.observability.fn_call_metrics.tracer", tracer):
        asyncio.run(test())


def test_TracingAsyncIterator_counts(local_tracing_context):
    tracer, exporter = local_tracing_context

    run_aiter(tracer, arange(10))

    assert (
        len(exporter.get_finished_spans())
        == 12  # 10 iteration spans + 1 extra end-of-stream span + 1 total wraping span
    )


def test_TracingAsyncIterator_has_iteration_counts(local_tracing_context):
    tracer, exporter = local_tracing_context

    run_aiter(tracer, arange(10))

    spans = exporter.get_finished_spans()

    parent_span = next(span for span in spans if span.parent is None)
    iteration_spans = [span for span in spans if span.parent is not None]

    assert parent_span is not None
    assert len(iteration_spans) > 0

    observed_iterations = []

    for span in iteration_spans:
        assert span.attributes["iteration"] is not None
        observed_iterations.append(span.attributes["iteration"])

    assert (
        list(set(observed_iterations)) == observed_iterations
    ), "No duplicate iterations"


def test_TracingAsyncIterator_arange_has_no_exception(
    local_tracing_context,
):
    tracer, exporter = local_tracing_context

    run_aiter(tracer, arange(10))

    spans: List[ReadableSpan] = exporter.get_finished_spans()

    event_spans = {event: span for span in spans for event in span.events}

    exception_events = [
        event for event in event_spans.keys() if event.name == "exception"
    ]

    assert exception_events == [], "Shouldn't have had any exceptions"


async def failing_async_iterator(stop: int, fails_after: int):
    for i in range(stop):
        yield i
        if i > fails_after:
            raise RuntimeError(f"Failed after {fails_after}")


def test_TracingAsyncIterator_has_the_right_exceptions(
    local_tracing_context,
):
    tracer, exporter = local_tracing_context

    try:
        run_aiter(tracer, failing_async_iterator(10, 5))
    except RuntimeError:
        pass

    spans: List[ReadableSpan] = exporter.get_finished_spans()

    event_spans = {event: span for span in spans for event in span.events}

    exception_events = [
        event
        for event, span in event_spans.items()
        if event.name == "exception" and span.name == "test iteration"
    ]

    assert len(exception_events) == 1
    assert exception_events[0].attributes
    assert exception_events[0].attributes["exception.message"] == "Failed after 5"
