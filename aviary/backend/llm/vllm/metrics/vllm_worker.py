from ray.util import metrics

from aviary.backend.observability.fn_call_metrics import (
    FnCallMetrics,
    FnCallMetricsContainer,
)

metrics_prefix = "vllm_worker"
worker_metrics = FnCallMetricsContainer(metrics_prefix)
num_input_tokens_gauge = metrics.Gauge(
    f"{metrics_prefix}_num_input_tokens",
    "Number of input tokens (without padding). Updated every iteration.",
)
num_seq_groups_gauge = metrics.Gauge(
    f"{metrics_prefix}_num_seq_groups",
    "Number of sequence groups (aka requests). Updated every iteration.",
)

model_metrics = FnCallMetrics("vllm_Model_forward")
sampler_metrics = FnCallMetrics("vllm_Sampler_forward")
