import collections
from typing import Any, List, Optional, Sequence

import ray._private.usage.usage_lib
from ray import serve

from rayllm.backend.llm.embedding.embedding_engine import EmbeddingEngine
from rayllm.backend.llm.embedding.embedding_models import EmbeddingApp
from rayllm.backend.llm.trtllm.trtllm_models import TRTLLMApp
from rayllm.backend.llm.vllm.vllm_engine import VLLMEngine
from rayllm.backend.llm.vllm.vllm_models import VLLMApp
from rayllm.backend.server.app import RouterDeployment
from rayllm.backend.server.embedding.embedding_deployment import EmbeddingDeployment
from rayllm.backend.server.models import EngineType, LLMApp, RouterArgs, ScalingConfig
from rayllm.backend.server.plugins.deployment_base_client import DeploymentBaseClient
from rayllm.backend.server.plugins.execution_hooks import ExecutionHooks
from rayllm.backend.server.plugins.multi_query_client import MultiQueryClient
from rayllm.backend.server.trtllm.trtllm_deployment import TRTLLMDeployment
from rayllm.backend.server.utils import parse_args
from rayllm.backend.server.vllm.vllm_deployment import VLLMDeployment


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
    try:
        bundles = scaling_config.as_placement_group_factory().bundles
    except ValueError:
        # May happen if all bundles are empty.
        bundles = []
    deployment_config["placement_group_bundles"] = [replica_actor_resources] + bundles
    deployment_config["placement_group_strategy"] = scaling_config.placement_strategy
    return deployment_config


def _clean_deployment_name(dep_name: str):
    return dep_name.replace("/", "--").replace(".", "_")


def get_deployment_name(app: LLMApp, name_prefix: str):
    return _clean_deployment_name(name_prefix + app.model_id)


def get_serve_deployment_args(app: LLMApp, name_prefix: str):
    deployment_config = set_deployment_placement_options(
        app.deployment_config.copy(deep=True).dict(), app.scaling_config  # type: ignore
    )

    # Set the name of the deployment config to map to the model id
    deployment_config["name"] = get_deployment_name(app, name_prefix)
    return deployment_config


def _get_execution_hooks():
    hooks = ExecutionHooks()
    return hooks


def get_llm_base_client(
    llm_base_models: Optional[Sequence[LLMApp]] = None, deployment_kwargs=None
):
    if not llm_base_models:
        return None

    base_configs = {model.model_id: model for model in llm_base_models}

    if deployment_kwargs is None:
        deployment_kwargs = {}

    base_deployments = {}
    for m in llm_base_models:
        if m.engine_config.type == EngineType.VLLMEngine:
            deployment_kwargs.setdefault("engine_cls", VLLMEngine)
            base_deployments[m.model_id] = VLLMDeployment.options(
                **get_serve_deployment_args(m, name_prefix="VLLMDeployment:")
            ).bind(base_config=m, **deployment_kwargs)
        elif m.engine_config.type == EngineType.EmbeddingEngine:
            deployment_kwargs.setdefault("engine_cls", EmbeddingEngine)
            base_deployments[m.model_id] = EmbeddingDeployment.options(
                **get_serve_deployment_args(m, name_prefix="EmbeddingDeployment:")
            ).bind(base_config=m, **deployment_kwargs)
        elif m.engine_config.type == EngineType.TRTLLMEngine:
            num_gpus = m.scaling_config.num_workers
            path = "rayllm.backend.llm.trtllm.trtllm_mpi.create_server"
            runtime_env = {
                "mpi": {
                    "args": ["-n", f"{int(num_gpus)}"],
                    "worker_entry": path,
                }
            }
            ray_actor_options = {"num_gpus": num_gpus, "runtime_env": runtime_env}
            deployment_config = {"ray_actor_options": ray_actor_options}
            if m.deployment_config:
                deployment_config = m.deployment_config.dict()
                if "ray_actor_options" in deployment_config:
                    deployment_config["ray_actor_options"].update(ray_actor_options)
                else:
                    deployment_config["ray_actor_options"] = ray_actor_options
            deployment_config["name"] = get_deployment_name(
                m, name_prefix="TRTLLMDeployment:"
            )

            base_deployments[m.model_id] = TRTLLMDeployment.options(
                **deployment_config,
            ).bind(base_config=m, **deployment_kwargs)
        else:
            raise ValueError(f"Unknown engine type {m.engine_config.type}")

    return DeploymentBaseClient(base_deployments, base_configs)


def get_embedding_base_client(
    embedding_models: Optional[List[EmbeddingApp]] = None, deployment_kwargs=None
):
    if not embedding_models:
        return None

    embedding_base_configs = {model.model_id: model for model in embedding_models}
    if not deployment_kwargs:
        deployment_kwargs = dict(engine_cls=EmbeddingEngine)
    embedding_base_deployments = {
        m.model_id: EmbeddingDeployment.options(
            **get_serve_deployment_args(m, name_prefix="EmbeddingDeployment:")
        ).bind(base_config=m, **deployment_kwargs)
        for m in embedding_models
    }
    embedding_base_client = DeploymentBaseClient(
        embedding_base_deployments, embedding_base_configs, model_type="embedding"
    )
    return embedding_base_client


def router_deployment(
    llm_base_models: List[LLMApp],
    enable_duplicate_models=False,
):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """
    if not enable_duplicate_models:
        ids = [
            model_deployment_config.engine_config.model_id
            for model_deployment_config in llm_base_models
        ]
        duplicate_models = {
            item for item, count in collections.Counter(ids).items() if count > 1
        }
        assert (
            not duplicate_models
        ), f"Found duplicate models {duplicate_models}. Please make sure all models have unique ids."

    hooks = _get_execution_hooks()

    llm_base_client = get_llm_base_client(llm_base_models)

    # Merged client
    merged_client = MultiQueryClient(llm_base_client, hooks=hooks)
    return RouterDeployment.bind(merged_client)


def router_application(args):
    ray._private.usage.usage_lib.record_library_usage("ray-llm")
    router_args = RouterArgs.parse_obj(args)
    vllm_apps = []
    embedding_apps = []
    trtllm_apps = []
    if router_args.models:
        ray._private.usage.usage_lib.record_library_usage("ray-llm-vllm")
        vllm_apps = parse_args(router_args.models, llm_app_cls=VLLMApp)
    if router_args.embedding_models:
        ray._private.usage.usage_lib.record_library_usage("ray-llm-embedding_models")
        embedding_apps = parse_args(
            router_args.embedding_models, llm_app_cls=EmbeddingApp
        )
    if router_args.trtllm_models:
        ray._private.usage.usage_lib.record_library_usage("ray-llm-tensorrt_llm")
        trtllm_apps = parse_args(router_args.trtllm_models, llm_app_cls=TRTLLMApp)
    return router_deployment(
        vllm_apps + embedding_apps + trtllm_apps, enable_duplicate_models=False
    )


def run(
    vllm_base_args: Optional[List[str]] = None,
    embedding_base_args: Optional[Any] = None,
    blocking: bool = False,
):
    """Run the LLM Server on the local Ray Cluster
    Args:
        models: The paths of the model yamls to deploy

    """
    assert (
        vllm_base_args or embedding_base_args
    ), "Neither vllm args or embedding args are provided."
    router_app = router_application(
        {"models": vllm_base_args, "embedding_models": embedding_base_args}
    )

    host = "0.0.0.0"

    serve.run(router_app, name="router", host=host, _blocking=blocking)

    deployment_address = f"http://{host}:8000"
    return deployment_address
