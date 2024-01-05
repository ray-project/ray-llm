import asyncio
from typing import AsyncIterator, Dict

from fastapi import HTTPException, Request
from ray import serve

from rayllm.backend.server.models import AviaryModelResponse, QueuePriority
from rayllm.backend.server.plugins.router_query_engine import (
    RouterQueryClient,
    StreamingErrorHandler,
)
from rayllm.backend.server.vllm.vllm_deployment import VLLMDeploymentImpl
from rayllm.common.models import DeletedModel, ModelData, Prompt
from rayllm.testing.mock_vllm_engine import MockVLLMEngine


class MockDeploymentImpl(VLLMDeploymentImpl):
    _default_engine_cls = MockVLLMEngine

    @staticmethod
    async def async_range(count):
        for i in range(count):
            yield (i)
            await asyncio.sleep(0.0)

    async def check_health(self):
        return True


@serve.deployment(
    # TODO make this configurable in aviary run
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
    },
    max_concurrent_queries=1,
)
class MockDeployment(MockDeploymentImpl):
    ...


class MockRouterQueryClient(RouterQueryClient):
    def __init__(self, mock_deployments: Dict[str, "MockDeployment"], hooks=None):
        self.mock_deployments = mock_deployments
        self._deployment_map = {}
        self.error_hook = StreamingErrorHandler(hooks=hooks)

    def _get_deploy_handle(self, model: str):
        deploy_handle = self.mock_deployments.get(model, None)
        if not deploy_handle:
            deploy_handle = self._deployment_map.get(model)
            if not deploy_handle:
                raise HTTPException(404, f"Could not find model with id {model}")
            deploy_handle = deploy_handle.options(stream=True)
            self.mock_deployments[model] = deploy_handle
        return deploy_handle

    async def stream(
        self, model: str, prompt: Prompt, request: Request, priority: QueuePriority
    ) -> AsyncIterator[AviaryModelResponse]:
        deploy_handle = self._get_deploy_handle(model)

        async for x in self.error_hook.handle_failure(
            model=model,
            request=request,
            prompt=prompt,
            async_iterator=deploy_handle.stream.remote("", prompt),
        ):
            yield x

    async def model(self, model_id: str) -> ModelData:
        """Get configurations for a supported model"""
        return ModelData(
            id=model_id,
            object="model",
            owned_by="mock owner",
            permission=["mock permission"],
            rayllm_metadata={
                "mock_metadata": "mock_metadata",
                "engine_config": {
                    "model_description": "mock_description",
                    "model_url": "mock_url",
                    "model_type": "text-generation",
                },
            },
        )

    async def models(self) -> Dict[str, ModelData]:
        """Get configurations for supported models"""
        metadatas = {}
        for model_id in self.mock_deployments:
            metadatas[model_id] = await self.model(model_id)
        return metadatas

    async def delete_fine_tuned_model(self, model: str) -> DeletedModel:
        if model not in self.mock_deployments:
            raise HTTPException(404, f"Could not find model with id {model}")
        del self.mock_deployments[model]
        return DeletedModel(id=model)
