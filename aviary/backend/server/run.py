import collections
from typing import List, Optional

import ray._private.usage.usage_lib
from ray import serve

from aviary.backend.llm.vllm.vllm_models import VLLMApp
from aviary.backend.server.app import RouterDeployment
from aviary.backend.server.models import LLMApp, ScalingConfig
from aviary.backend.server.plugins.deployment_base_client import DeploymentBaseClient
from aviary.backend.server.plugins.execution_hooks import (
    ExecutionHooks,
)
from aviary.backend.server.plugins.multi_query_client import MultiQueryClient
from aviary.backend.server.utils import parse_args
from aviary.backend.server.vllm.vllm_deployment import VLLMDeployment


def set_deployment_placement_options(
    deployment_config: dict, scaling_config: ScalingConfig
):
    scaling_config = scaling_config.as_air_scaling_config()
    deployment_config.setdefault("ray_actor_options", {})
    replica_actor_resources = {
        "CPU": deployment_config["ray_actor_options"].get("num_cpus", 1),
        "GPU": deployment_config["ray_actor_options"].get("num_gpus", 0),
        **deployment_config["ray_actor_options"].get("resources", {}),
    }
    if (
        "placement_group_bundles" in deployment_config
        or "placement_group_strategy" in deployment_config
    ):
        raise ValueError(
            "placement_group_bundles and placement_group_strategy must not be specified in deployment_config. "
            "Use scaling_config to configure replicaplacement group."
        )
    deployment_config["placement_group_bundles"] = [
        replica_actor_resources
    ] + scaling_config.as_placement_group_factory().bundles
    deployment_config["placement_group_strategy"] = scaling_config.placement_strategy
    return deployment_config


def _clean_deployment_name(dep_name: str):
    return dep_name.replace("/", "--").replace(".", "_")


def get_serve_deployment_args(app: LLMApp, name_prefix: str):
    deployment_config = set_deployment_placement_options(
        app.deployment_config.copy(deep=True).dict(), app.scaling_config  # type: ignore
    )

    # Set the name of the deployment config to map to the model id
    deployment_config["name"] = _clean_deployment_name(name_prefix + app.model_id)
    return deployment_config


def _get_execution_hooks():
    hooks = ExecutionHooks()
    return hooks


def get_vllm_base_client(vllm_base_models: Optional[List[VLLMApp]] = None):
    if not vllm_base_models:
        return None

    vllm_base_configs = {model.model_id: model for model in vllm_base_models}
    vllm_base_deployments = {
        m.model_id: VLLMDeployment.options(
            **get_serve_deployment_args(m, name_prefix="VLLMDeployment:")
        ).bind(m)
        for m in vllm_base_models
    }
    vllm_base_client = DeploymentBaseClient(vllm_base_deployments, vllm_base_configs)
    return vllm_base_client


def router_deployment(
    vllm_base_models: List[LLMApp],
    enable_duplicate_models=False,
):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """
    if not enable_duplicate_models:
        ids = [
            model_deployment_config.engine_config.model_id
            for model_deployment_config in vllm_base_models
        ]
        duplicate_models = {
            item for item, count in collections.Counter(ids).items() if count > 1
        }
        assert (
            not duplicate_models
        ), f"Found duplicate models {duplicate_models}. Please make sure all models have unique ids."

    hooks = _get_execution_hooks()

    vllm_base_client = get_vllm_base_client(vllm_base_models)

    # Get all clients that were created
    clients = [vllm_base_client]
    clients = [x for x in clients if x]

    # Merged client
    merged_client = MultiQueryClient(*clients, hooks=hooks)
    return RouterDeployment.bind(merged_client)


def router_application(args):
    llm_apps = parse_args(args, llm_app_cls=VLLMApp)
    return router_deployment(llm_apps, enable_duplicate_models=False)


def run(
    models: List[str],
    blocking: bool = False,
):
    """Run the LLM Server on the local Ray Cluster
    Args:
        models: The paths of the model yamls to deploy

    """
    ray._private.usage.usage_lib.record_library_usage("aviary")
    router_app = router_application(models)

    serve.run(
        router_app, name="router", route_prefix="/", host="0.0.0.0", _blocking=blocking
    )
