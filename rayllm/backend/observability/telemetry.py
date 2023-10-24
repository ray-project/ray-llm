import asyncio
import logging
import time

from fastapi import FastAPI
from ray.util import metrics
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from rayllm.backend.observability.event_loop_monitoring import (
    _LATENCY_HISTOGRAM_BOUNDARIES,
    setup_event_loop_monitoring,
)
from rayllm.backend.observability.tracing.fastapi import (
    FastAPIInstrumentor,
    _get_route_details,
)

logger = logging.getLogger(__name__)


def configure_telemetry(app: FastAPI, name: str):
    """Configures common telemetry hooks for FastAPI applications"""

    # Would prefer to patch all of FastAPI, but i don't have enough control over the
    # startup sequence of processes to make this work right now.
    FastAPIInstrumentor().instrument_app(app)

    app.state.name = name

    register_event_loop_telemetry(app)
    register_http_metrics_middleware(app)


def register_event_loop_telemetry(app: FastAPI):
    @app.on_event("startup")
    async def add_fastapi_event_loop_monitoring():
        # Store the task handle to prevent it from being garbage collected
        app.state.fastapi_event_loop_schedule_latency_metrics = metrics.Histogram(
            "anyscale_fastapi_event_loop_schedule_latency",
            description="Latency of getting yielded control on the FastAPI event loop in seconds",
            boundaries=_LATENCY_HISTOGRAM_BOUNDARIES,
            tag_keys=("api_server",),
        )
        app.state.fastapi_event_loop_monitoring_iterations = metrics.Counter(
            "anyscale_fastapi_event_loop_monitoring_iterations",
            description="Number of times the FastAPI event loop has iterated to get anyscale_fastapi_event_loop_schedule_latency.",
            tag_keys=("api_server",),
        )
        app.state.fastapi_event_loop_monitoring_tasks = metrics.Gauge(
            "anyscale_fastapi_event_loop_monitoring_tasks",
            description="Number of outsanding tasks on the FastAPI event loop.",
            tag_keys=("api_server",),
        )

        tags = {"api_server": _get_app_name(app)}

        app.state.fastapi_event_loop_schedule_latency_metrics_task = (
            setup_event_loop_monitoring(
                asyncio.get_running_loop(),
                app.state.fastapi_event_loop_schedule_latency_metrics,
                app.state.fastapi_event_loop_monitoring_iterations,
                app.state.fastapi_event_loop_monitoring_tasks,
                tags,
            )
        )


def register_http_metrics_middleware(app: FastAPI):
    @app.on_event("startup")
    def setup_http_metrics() -> None:
        app.state.http_requests_metrics = metrics.Counter(
            "anyscale_http_requests",
            description=(
                "Total number of HTTP requests by status code, handler and method."
            ),
            tag_keys=("code", "handler", "method", "api_server"),
        )
        app.state.http_requests_latency_metrics = metrics.Histogram(
            "anyscale_http_request_duration_seconds",
            description=("Duration in seconds of HTTP requests."),
            boundaries=[
                0.01,
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                1,
                1.5,
                2,
                5,
                10,
                30,
                60,
                120,
                300,
            ],
            tag_keys=("handler", "method", "api_server"),
        )

    @app.middleware("http")
    async def add_http_requests_metrics(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        now = time.monotonic()
        resp = await call_next(request)
        duration = time.monotonic() - now

        handler = _get_route_details(request.scope) or "unknown"
        method = request.get("method", "UNKNOWN")

        if hasattr(app.state, "http_requests_metrics"):
            app.state.http_requests_metrics.inc(
                # increment count by 1
                1,
                {
                    "code": str(resp.status_code),
                    "handler": handler,
                    "api_server": _get_app_name(app),
                    "method": method,
                },
            )
        else:
            logger.debug("HTTP requests telemetry not initialized, skipping")

        if hasattr(app.state, "http_requests_latency_metrics"):
            app.state.http_requests_latency_metrics.observe(
                duration,
                {
                    "handler": handler,
                    "api_server": _get_app_name(app),
                    "method": method,
                },
            )
        else:
            logger.debug("HTTP requests telemetry not initialized, skipping")

        return resp


def _get_app_name(app: FastAPI) -> str:
    if hasattr(app.state, "name"):
        return app.state.name

    return "unknown"
