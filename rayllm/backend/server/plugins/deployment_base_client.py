import logging
from typing import Dict

from fastapi import HTTPException, Request
from ray.serve.handle import DeploymentHandle

from rayllm.backend.server.models import LLMApp, QueuePriority
from rayllm.backend.server.plugins.router_query_engine import (
    RouterQueryClient,
)
from rayllm.common.models import ModelData, Prompt

logger = logging.getLogger(__name__)


class DeploymentBaseClient(RouterQueryClient):
    """A RouterQueryClient that queries a Serve Deployment.

    Deployment map is a map from model id to deployment handle,
    which is used to resolve the deployment to query.
    """

    def __init__(
        self,
        deployment_map: Dict[str, DeploymentHandle],
        engine_configs: Dict[str, LLMApp],
    ):
        logger.info(f"Initialized with base handles {deployment_map}")
        self._deployment_map = deployment_map
        self._engine_configurations = engine_configs
        self.all_models = {
            key: _model_def(val) for key, val in self._engine_configurations.items()
        }

    async def stream(
        self, model: str, prompt: Prompt, request: Request, priority: QueuePriority
    ):
        request_id = request.state.request_id
        deploy_handle = self._get_deploy_handle(model)
        async for x in deploy_handle.options(stream=True).stream.remote(
            request_id, prompt
        ):
            yield x

    def _get_deploy_handle(self, model: str):
        deploy_handle = self._deployment_map.get(model)
        if not deploy_handle:
            raise HTTPException(404, f"Could not find model with id {model}")
        return deploy_handle

    async def models(self):
        return self.all_models

    async def model(self, model: str):
        return self.all_models.get(model)


def _model_def(app: LLMApp):
    metadata = app.dict(
        include={
            "engine_config": {
                "generation",
                "model_id",
                "model_url",
                "model_description",
            }
        }
    )
    return ModelData(
        id=app.model_id,
        object="model",
        owned_by="organization-owner",  # TODO
        permission=[],  # TODO
        aviary_metadata=metadata,
    )
