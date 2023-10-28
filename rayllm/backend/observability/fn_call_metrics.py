import inspect
import time
from contextlib import AbstractContextManager, contextmanager
from enum import Enum
from functools import wraps
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

from opentelemetry import context, trace
from ray.util import metrics

from rayllm.backend.observability.metrics import NonExceptionThrowingCounter as Counter

tracer = trace.get_tracer(__name__)


# Histogram buckets for short-range latencies measurements:
# Min=1ms, Max=30s
#
# NOTE: Number of buckets have to be bounded (and not exceed 30)
#       to avoid overloading metrics sub-system
SHORT_RANGE_LATENCY_HISTOGRAM_BUCKETS_MS: List[float] = [
    1,
    2,
    5,
    10,
    15,
    20,
    30,
    40,
    50,
    60,
    75,
    100,
    125,
    150,
    175,
    200,
    300,
    400,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    7500,
    10000,
    20000,
    30000,
]

# Histogram buckets for long-range latencies measurements:
# Min=10ms, Max=300s
LONG_RANGE_LATENCY_HISTOGRAM_BUCKETS_MS = [
    x * 10 for x in SHORT_RANGE_LATENCY_HISTOGRAM_BUCKETS_MS
]


class ClockUnit(int, Enum):
    ms = 1000
    s = 1


class MsClock:
    """A clock that tracks intervals in milliseconds"""

    def __init__(self, unit: ClockUnit = ClockUnit.ms):
        self.reset()
        self.unit = unit

    def reset(self):
        self.start_time = time.perf_counter()

    def interval(self):
        return (time.perf_counter() - self.start_time) * self.unit

    def reset_interval(self):
        interval = self.interval()
        self.reset()
        return interval


T = TypeVar("T")


class FnCallMetrics:
    """Instrument a function to get call metrics

    This class combines 3 OpenTelemetry metrics. When used to instrument a function,
    it tracks the number of times a function is called, the latency and count for successes,
    and the latency and count for failures.

    Usage:
    metrics = FnCallMetrics(...)

    # this will record open telemetry metrics for your function
    @metrics.wrap
    def my_function(self):
        ...
    """

    def __init__(
        self,
        prefix: str,
        description: str = "",
        tag_keys: Optional[Sequence[str]] = None,
        latency_histogram_buckets: Optional[List[float]] = None,
    ):
        self.prefix = prefix
        self.description_prefix = description

        self.num_started = Counter(
            self._suffix("total"),
            description=self._description("Total number of calls"),
            tag_keys=tag_keys,
        )

        target_histogram_buckets = (
            latency_histogram_buckets or SHORT_RANGE_LATENCY_HISTOGRAM_BUCKETS_MS
        )

        self.success_latency = metrics.Histogram(
            self._suffix("succeeded_latency_ms"),
            description=self._description("Latency of succeeded calls"),
            boundaries=target_histogram_buckets,
            tag_keys=tag_keys,
        )
        self.failure_latency = metrics.Histogram(
            self._suffix("failed_latency_ms"),
            description=self._description("Latency of failed calls"),
            boundaries=target_histogram_buckets,
            tag_keys=tag_keys,
        )

    def _suffix(self, name: str):
        return f"{self.prefix}_{name}"

    def _description(self, suffix: str):
        return f"{self.description_prefix} -- {suffix}"

    @contextmanager
    def record(self, **tags):
        """Manage the incrementing of counters and recording of latency."""
        tags = tags or {}
        clock = MsClock()
        try:
            self.num_started.inc(tags=tags)
            yield
            # Record the success latency
            self.success_latency.observe(clock.interval(), tags=tags)
        except BaseException:
            # Record the error latency
            self.failure_latency.observe(clock.interval(), tags=tags)
            # Raise the error
            raise

    def wrap(self, wrapped: Callable[..., T]) -> Callable[..., T]:
        """Provides an API to decorate methods enabling metrics recording"""

        if inspect.iscoroutinefunction(wrapped):

            @wraps(wrapped)
            async def async_wrapper(*args, **kwargs):
                with self.record():
                    return await wrapped(*args, **kwargs)

            return async_wrapper

        else:

            @wraps(wrapped)
            def wrapper(*args, **kwargs):
                with self.record():
                    return wrapped(*args, **kwargs)

            return wrapper


class FnCallMetricsContainer:
    """Container for multiple FnCallMetrics objects intended
    to be used for one class. Will automatically derive
    names for the metrics based on the class/function name.
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.metrics: Dict[str, FnCallMetrics] = {}

    def wrap(self, wrapped: Callable[..., T]) -> Callable[..., T]:
        self.metrics[wrapped.__name__] = FnCallMetrics(
            f"{self.prefix}_{wrapped.__name__}"
        )
        return self.metrics[wrapped.__name__].wrap(wrapped)


class InstrumentTokenAsyncGenerator:
    """This class instruments an asynchronous generator.

    It gathers 3 metrics:
    1. Time to first time
    2. Time between tokens
    3. Total completion time

    Usage:

    @InstrumentTokenAsyncGenerator("my_special_fn")
    async def to_instrument():
        yield ...
    """

    all_instrument_names: Set[str] = set()

    def __init__(
        self, generator_name: str, latency_histogram_buckets: List[float] = None
    ):
        self.generator_name = f"aviary_{generator_name}"

        target_latency_histogram_buckets = (
            latency_histogram_buckets or SHORT_RANGE_LATENCY_HISTOGRAM_BUCKETS_MS
        )

        assert (
            self.generator_name not in self.all_instrument_names
        ), "This generator name was already used elsewhere. Please specify another one."
        self.all_instrument_names.add(self.generator_name)

        self.token_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_per_token_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=target_latency_histogram_buckets,
        )

        self.first_token_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_first_token_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=target_latency_histogram_buckets,
        )
        self.total_latency_histogram = metrics.Histogram(
            f"{self.generator_name}_total_latency_ms",
            f"Generator metrics for {self.generator_name}",
            boundaries=target_latency_histogram_buckets,
        )

    def __call__(
        self, async_generator_fn: Callable[..., AsyncGenerator[T, None]]
    ) -> Callable[..., AsyncGenerator[T, None]]:
        async def new_gen(*args, **kwargs):
            interval_clock = MsClock()
            total_clock = MsClock()
            is_first_token = True
            try:
                with TracingAsyncIterator(
                    self.generator_name, async_generator_fn(*args, **kwargs)
                ) as tracing_iterator:
                    async for x in tracing_iterator:
                        if is_first_token:
                            tracing_iterator.span.add_event("first_token")

                            self.first_token_latency_histogram.observe(
                                total_clock.interval()
                            )
                            interval_clock.reset()
                            is_first_token = False
                        else:
                            self.token_latency_histogram.observe(
                                interval_clock.reset_interval()
                            )
                        yield x
            finally:
                self.total_latency_histogram.observe(total_clock.interval())

        return new_gen


class TracingAsyncIterator:
    span: trace.Span
    span_context: context.Context

    iterations: int = 0
    is_done: bool = False

    def __init__(self, name: str, wrapped: AsyncIterator):
        self.wrapped = wrapped

        self.name = name
        self.span_context_manager: AbstractContextManager = (
            tracer.start_as_current_span(self.name)
        )

    def __enter__(self):
        self.span = self.span_context_manager.__enter__()
        self.span_context = context.get_current()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.span_context_manager.__exit__(exc_type, exc_value, traceback)

    def __aiter__(self):
        return self

    async def __anext__(self):
        # Note for future readers: this is actually slightly confusing as there
        # will always be one too many iterations because we don't know the
        # __anext__ is going to throw a StopAsyncIteration exception, until
        # after we start the span. And unfortunately, I can't find a way to bail
        # on the span after we start it.
        with tracer.start_as_current_span(
            f"{self.name} iteration",
            context=self.span_context,
            attributes={"iteration": self.iterations},
        ):
            try:
                return await self.wrapped.__anext__()
            except StopAsyncIteration:
                self.is_done = True
            finally:
                self.iterations += 1

        if self.is_done:
            # This exists because we need to throw this exception from outside
            # of the tracer.start_as_current_span context. Otherwise it'll get recorded
            # as a part of the span and we don't need to mark the span as an
            # error because we got a StopIterationException.
            raise StopAsyncIteration
