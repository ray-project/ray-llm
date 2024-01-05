import logging
from typing import Dict, Optional

from fastapi import HTTPException, Request, status
from ray.serve.handle import DeploymentHandle

from rayllm.backend.server.models import LLMApp, QueuePriority
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.plugins.router_query_engine import (
    RouterQueryClient,
)
from rayllm.common.models import DeletedModel, ModelData, Prompt

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
        model_type: Optional[str] = None,
    ):
        logger.info(f"Initialized with base handles {deployment_map}")
        self.model_type = model_type
        self._deployment_map = deployment_map
        self._deploy_handles: Dict[str, DeploymentHandle] = {}
        self._engine_configurations = engine_configs
        self.all_models = {
            key: _model_def(val) for key, val in self._engine_configurations.items()
        }

    async def stream(
        self, model: str, prompt: Prompt, request: Request, priority: QueuePriority
    ):
        request_id = request.state.request_id
        self._update_request_attributes(request)
        deploy_handle = self._get_deploy_handle(model)

        async for x in deploy_handle.stream.remote(request_id, prompt):
            yield x

    def _get_deploy_handle(self, model: str):
        # This ensures we only have one .options(stream=True) handle
        # per model (fixes metrics cardinality)
        deploy_handle = self._deploy_handles.get(model, None)
        if not deploy_handle:
            deploy_handle = self._deployment_map.get(model)
            if not deploy_handle:
                raise HTTPException(404, f"Could not find model with id {model}")
            deploy_handle = deploy_handle.options(stream=True)
            self._deploy_handles[model] = deploy_handle
        return deploy_handle

    async def models(self):
        return self.all_models

    async def model(self, model: str):
        return self.all_models.get(model)

    async def delete_fine_tuned_model(self, model: str) -> DeletedModel:
        raise OpenAIHTTPException(
            status.HTTP_403_FORBIDDEN, f"Cannot delete base model with id {model}"
        )

    def _update_request_attributes(self, request: Request):
        request.state.billing_attributes = getattr(
            request.state, "billing_attributes", {}
        )

        if self.model_type is not None:
            request.state.billing_attributes["model_type"] = self.model_type


def _model_def(app: LLMApp):
    metadata = app.short_metadata()
    return ModelData(
        id=app.model_id,
        object="model",
        owned_by="organization-owner",  # TODO
        permission=[],  # TODO
        rayllm_metadata=metadata,
    )
