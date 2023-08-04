import os
import sys
import time
from typing import Dict, List, Tuple, Union

import ray._private.usage.usage_lib
from ray import serve
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import ServeInstanceDetails

from aviary.backend.server.app import (
    RouterDeployment,
    TextGenerationInferenceLLMDeployment,
)
from aviary.backend.server.models import (
    AppArgs,
    LLMApp,
    RouterArgs,
    TextGenerationInferenceEngineConfig,
)
from aviary.backend.server.utils import parse_args
from aviary.conf import ENV_VARS_TO_PROPAGATE


def llm_model(app: LLMApp):
    print("Initializing LLM app", app.json(indent=2))
    user_config = app.dict()
    deployment_config = app.deployment_config.dict()
    deployment_config = deployment_config.copy()

    if isinstance(app.engine_config, TextGenerationInferenceEngineConfig):
        deployment_cls = TextGenerationInferenceLLMDeployment
        max_concurrent_queries = deployment_config.pop("max_concurrent_queries", None)
        if max_concurrent_queries is None:
            raise ValueError(
                "deployment_config.max_concurrent_queries must be specified for continuous batching models."
            )

    deployment_config.setdefault("ray_actor_options", {})
    deployment_config["ray_actor_options"].setdefault("runtime_env", {})
    deployment_config["ray_actor_options"]["runtime_env"].setdefault("env_vars", {})
    for env_var in ENV_VARS_TO_PROPAGATE:
        if env_var in os.environ:
            deployment_config["ray_actor_options"]["runtime_env"]["env_vars"][
                env_var
            ] = os.getenv(env_var)

    return deployment_cls.options(
        name=app.engine_config.model_id.replace("/", "--").replace(".", "_"),
        max_concurrent_queries=max_concurrent_queries,
        user_config=user_config,
        **deployment_config,
    ).bind()


def _parse_config_for_router(
    engine_config: TextGenerationInferenceEngineConfig,
) -> Tuple[str, str, str]:
    deployment_name = engine_config.model_id.replace("/", "--").replace(".", "_")
    deployment_route = f"/{deployment_name}"
    full_deployment_name = f"{deployment_name}_{deployment_name}"
    return deployment_name, deployment_route, full_deployment_name


def router_deployment(models: Dict[str, Union[str, LLMApp]]):
    app_names = {}
    deployment_routes = {}
    full_deployment_names = {}
    engine_configs = {}
    for id, model in models.items():
        model = parse_args(model)[0]
        engine_configs[id] = model
        (
            deployment_name,
            deployment_route,
            full_deployment_name,
        ) = _parse_config_for_router(model.engine_config)
        app_name = deployment_name
        app_names[model.engine_config.model_id] = app_name
        deployment_routes[model.engine_config.model_id] = deployment_route
        full_deployment_names[model.engine_config.model_id] = full_deployment_name

    router_deployment = RouterDeployment.bind(
        full_deployment_names, deployment_routes, engine_configs
    )
    return router_deployment


def llm_server(args: Union[str, LLMApp, List[Union[LLMApp, str]]]):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    models = parse_args(args)
    if not models:
        raise RuntimeError("No enabled models were found.")

    # For each model, create a deployment
    deployments = {}
    app_names = {}
    deployment_routes = {}
    full_deployment_names = {}
    engine_configs = {}
    for model in models:
        if model.engine_config.model_id in engine_configs:
            raise ValueError(
                f"Duplicate model_id {model.engine_config.model_id} specified. "
                "Please ensure that each model has a unique model_id. "
                "If you want two models to share the same Hugging Face Hub ID, "
                "specify initialization.hf_model_id in the model config."
            )
        engine_configs[model.engine_config.model_id] = model
        deployments[model.engine_config.model_id] = llm_model(model)

        (
            deployment_name,
            deployment_route,
            full_deployment_name,
        ) = _parse_config_for_router(model.engine_config)
        app_name = deployment_name
        app_names[model.engine_config.model_id] = app_name
        deployment_routes[model.engine_config.model_id] = deployment_route
        full_deployment_names[model.engine_config.model_id] = full_deployment_name

    router = router_deployment(engine_configs)

    return router, deployments, deployment_routes, app_names


def llm_application(args):
    """This is a simple wrapper for LLM Server
    That is compatible with the yaml config file format

    """
    serve_args = AppArgs.parse_obj(args)
    model = parse_args(serve_args.model)[0]
    return llm_model(model)


def router_application(args):
    serve_args = RouterArgs.parse_obj(args)
    return router_deployment(serve_args.models)


def _all_applications_healthy():
    address = os.environ.get("RAY_AGENT_ADDRESS", "http://localhost:52365")
    serve_status = ServeInstanceDetails(
        **ServeSubmissionClient(address).get_serve_details()
    )
    if any(
        app.status == "DEPLOY_FAILED" for name, app in serve_status.applications.items()
    ):
        raise RuntimeError(
            "One or more applications failed to deploy. "
            "Check output above and Ray Dashboard/Ray logs for more details."
        )
    return all(
        app.status == "RUNNING" for name, app in serve_status.applications.items()
    )


def run(*models: Union[LLMApp, str], blocking: bool = True):
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    router_app, deployments, deployment_routes, app_names = llm_server(list(models))

    ray._private.usage.usage_lib.record_library_usage("aviary")

    for model_id in deployments.keys():
        app = deployments[model_id]
        route = deployment_routes[model_id]
        app_name = app_names[model_id]
        serve.run(
            app, name=app_name, route_prefix=route, host="0.0.0.0", _blocking=False
        )

    serve.run(
        router_app, name="router", route_prefix="/", host="0.0.0.0", _blocking=False
    )

    if blocking:
        while not _all_applications_healthy():
            time.sleep(1)


if __name__ == "__main__":
    run(*sys.argv[1:])
