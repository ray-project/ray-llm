from ray import serve

from rayllm.backend.llm.vllm.vllm_models import VLLMApp
from rayllm.backend.server.app import RouterDeployment
from rayllm.testing.mock_deployment import MockDeployment, MockRouterQueryClient

vllm_app_def = """
deployment_config:
  autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_num_ongoing_requests_per_replica: 5
    metrics_interval_s: 10.0
    look_back_period_s: 30.0
    smoothing_factor: 1.0
    downscale_delay_s: 300.0
    upscale_delay_s: 60.0
  max_concurrent_queries: 15
  ray_actor_options:
    resources:
      accelerator_type_a10: 0
multiplex_config:
    max_num_models_per_replica: 16
engine_config:
  model_id: meta-llama/Llama-2-7b-hf
  type: VLLMEngine
  engine_kwargs:
    trust_remote_code: True
  max_total_tokens: 4096
  generation:
    prompt_format:
      system: "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n"
      assistant: " {instruction} </s><s> [INST] "
      trailing_assistant: " "
      user: "{instruction} [/INST]"
      default_system_message: "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share False information."
    stopping_sequences: []
scaling_config:
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 8
  placement_strategy: "STRICT_PACK"
  resources_per_worker:
    accelerator_type_a10: 0
"""  # noqa


def router_application(num_mock_deployments: int = 1, hooks=None):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """

    deployment_map = {}

    for i in range(num_mock_deployments):
        deployment_map[f"model_{i}"] = MockDeployment.options(
            name=f"MockDeployment:model_{i}",
        ).bind(
            VLLMApp.parse_yaml(vllm_app_def),
        )

    # Merged client
    merged_client = MockRouterQueryClient(deployment_map, hooks=hooks)
    return RouterDeployment.options(
        autoscaling_config={
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 1,
        }
    ).bind(merged_client)


def run(num_mock_deployments: int = 1, route_prefix="/", hooks=None):
    """Run the Mock LLM Server on the local Ray Cluster
    Args:
        num_mock_deployments: Number of mock deployments to run

    """
    router_app = router_application(num_mock_deployments, hooks=hooks)

    serve.run(
        router_app,
        name="router",
        route_prefix=route_prefix,
        host="0.0.0.0",
        _blocking=True,
    )


if __name__ == "__main__":
    run()
