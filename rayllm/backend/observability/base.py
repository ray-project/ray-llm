import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Optional

from opentelemetry import trace
from opentelemetry.util.types import AttributeValue

from rayllm.backend.observability.tracing.baggage import baggage as set_baggage

# Standard loggers
serve_logger = logging.getLogger("ray.serve")
aviary_logger = logging.getLogger("aviary")

# If the AVIARY_DEBUG flag is set, then set the log level to debug
if os.getenv("AVIARY_DEBUG"):
    aviary_logger.setLevel(logging.DEBUG)

# Standard tracers
tracer = trace.get_tracer(__name__)


@contextmanager
def step(
    step_name: str,
    request_id: Optional[str] = None,
    attrs: Optional[Dict[str, AttributeValue]] = None,
    baggage: Optional[Dict[str, AttributeValue]] = None,
):
    if baggage is None:
        baggage = {}

    # TODO(tchordia): Add tracing here
    t = time.monotonic()
    try:
        aviary_logger.debug(f"Starting {step_name} at {t}. Id: {request_id}")
        with set_baggage(baggage), tracer.start_as_current_span(
            step_name, attributes=attrs
        ) as span:
            yield span
    finally:
        took = time.monotonic() - t
        aviary_logger.debug(f"Completed {step_name}, took {took}s. Id: {request_id}")
