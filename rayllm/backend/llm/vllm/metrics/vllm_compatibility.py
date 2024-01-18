from typing import Dict

from ray.util import metrics

from rayllm.backend.observability.fn_call_metrics import FnCallMetricsContainer

# TODO (avnishn, yard1): move this all into vllm eventually

metrics_prefix = "vllm_llm_engine"
engine_metrics = FnCallMetricsContainer(metrics_prefix)
# The following metrics are sampled only every 5s due to the overhead of
# collecting them and the fact that the logic to collect them is coming
# from vLLM directly. We may want to revisit this in the future if
# the metrics need more granularity.
engine_record_stats_gauges: Dict[str, metrics.Gauge] = {
    "avg_prompt_throughput": metrics.Gauge(
        f"{metrics_prefix}_avg_prompt_throughput",
        "avg prompt (prefill) throughput (tokens/s). Updated every 5s.",
    ),
    "avg_generation_throughput": metrics.Gauge(
        f"{metrics_prefix}_avg_generation_throughput",
        "avg generation throughput (tokens/s). Updated every 5s.",
    ),
    "gpu_cache_usage": metrics.Gauge(
        f"{metrics_prefix}_gpu_cache_usage", "gpu_cache_usage (%). Updated every 5s."
    ),
    "cpu_cache_usage": metrics.Gauge(
        f"{metrics_prefix}_cpu_cache_usage", "cpu_cache_usage (%). Updated every 5s."
    ),
}

running_requests_gauge = metrics.Gauge(
    f"{metrics_prefix}_running_requests", "running requests. Updated every iteration."
)
swapped_requests_gauge = metrics.Gauge(
    f"{metrics_prefix}_swapped_requests", "swapped requests. Updated every iteration."
)
waiting_requests_gauge = metrics.Gauge(
    f"{metrics_prefix}_waiting_requests", "waiting requests. Updated every iteration."
)
