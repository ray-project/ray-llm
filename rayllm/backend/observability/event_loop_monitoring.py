import asyncio
import logging
import time
from typing import Dict

from ray.util import metrics

logger = logging.getLogger(__name__)

_METRICS_LOOP_INTERVAL = 5  # 5 seconds
_LATENCY_HISTOGRAM_BOUNDARIES = [
    0.05,
    0.1,
    0.15,
    0.20,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    45.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
    300.0,
    600.0,
]


def setup_event_loop_monitoring(
    loop: asyncio.AbstractEventLoop,
    scheduling_latency: metrics.Histogram,
    iterations: metrics.Counter,
    tasks: metrics.Gauge,
    tags: Dict[str, str],
) -> asyncio.Task:
    return asyncio.create_task(
        _run_monitoring_loop(
            loop,
            schedule_latency=scheduling_latency,
            iterations=iterations,
            task_gauge=tasks,
            tags=tags,
        )
    )


async def _run_monitoring_loop(
    loop: asyncio.AbstractEventLoop,
    schedule_latency: metrics.Histogram,
    iterations: metrics.Counter,
    task_gauge: metrics.Gauge,
    tags: Dict[str, str],
) -> None:
    while loop.is_running():
        iterations.inc(1, tags)
        num_tasks = len(asyncio.all_tasks())
        task_gauge.set(num_tasks, tags)
        yield_time = time.monotonic()
        await asyncio.sleep(_METRICS_LOOP_INTERVAL)
        elapsed_time = time.monotonic() - yield_time

        # Historically, Ray's implementation of histograms are extremely finicky
        # with non-positive values (https://github.com/ray-project/ray/issues/26698).
        # Technically it shouldn't be possible for this to be negative, add the
        # max just to be safe.
        latency = max(0, elapsed_time - _METRICS_LOOP_INTERVAL)
        schedule_latency.observe(latency, tags)


def _event_loop_available() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        # Likely that actor is being run outside of Ray, for example in a
        # unit test.
        return False


class InstrumentAsyncioEventLoop:
    """
    This is a mixin that starts an asyncio task to monitors the health of the
    event loop. This is meant to be added to any Actor class that has an asyncio
    reconciler.
    """

    def __init__(self, *args, **kwargs):
        # This calls other parent class __init__ methods in multiple inheritance
        # situations
        super().__init__(*args, **kwargs)

        if _event_loop_available():
            tag_keys = ("actor",)

            iterations = metrics.Counter(
                "anyscale_event_loop_monitoring_iterations",
                description="Number of times the monitoring loop has iterated for this actor.",
                tag_keys=tag_keys,
            )
            tasks = metrics.Gauge(
                "anyscale_event_loop_tasks",
                description="Number of outstanding tasks on the event loop.",
                tag_keys=tag_keys,
            )
            scheduling_latency = metrics.Histogram(
                "anyscale_event_loop_schedule_latency",
                description="Latency of getting yielded control on the event loop in seconds",
                boundaries=_LATENCY_HISTOGRAM_BOUNDARIES,
                tag_keys=tag_keys,
            )

            tags = {"actor": self.__class__.__name__}

            # Store the task handle to prevent it from being garbage collected
            self._event_loop_metrics_task = setup_event_loop_monitoring(
                asyncio.get_running_loop(),
                scheduling_latency=scheduling_latency,
                iterations=iterations,
                tasks=tasks,
                tags=tags,
            )
        else:
            logger.info("No event loop running. Skipping event loop metrics.")
