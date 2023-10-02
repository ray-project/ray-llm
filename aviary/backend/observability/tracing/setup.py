import os
import secrets
import socket

from opentelemetry import (
    trace,
)
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util._once import Once

from aviary.backend.observability.tracing.baggage_span_processor import (
    BaggageSpanProcessor,
)
from aviary.backend.observability.tracing.fastapi import FastAPIInstrumentor
from aviary.backend.observability.tracing.threading import ThreadingInstrumentor

has_setup_tracing = Once()


def _setup_tracing():
    tracer_provider = TracerProvider(
        resource=Resource(
            attributes={
                "service.name": "aviary",
                "meta.local_hostname": socket.gethostname(),
                "meta.process_id": os.getpid(),
                "meta.process_unique_id": secrets.token_urlsafe(),
            }
        ),
    )

    tracer_provider.add_span_processor(  # type: ignore
        # Ensure we set baggage entries as attributes on all spans
        BaggageSpanProcessor()
    )

    trace.set_tracer_provider(tracer_provider)

    # Not really effective right now, still need to `instrument_app` each app
    FastAPIInstrumentor().instrument()

    AioHttpClientInstrumentor().instrument()
    BotocoreInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()
    ThreadingInstrumentor().instrument()
    RedisInstrumentor().instrument()

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("startup"):
        pass


def setup_tracing() -> None:
    has_setup_tracing.do_once(_setup_tracing)
