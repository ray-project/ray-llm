from typing import AsyncIterator, Dict, Optional

from fastapi import status
from starlette.requests import Request

from rayllm.backend.logger import get_logger
from rayllm.backend.server.metrics import Metrics
from rayllm.backend.server.models import (
    Args,
    AviaryModelResponse,
    Prompt,
    QueuePriority,
)
from rayllm.backend.server.openai_compat.openai_exception import OpenAIHTTPException
from rayllm.backend.server.plugins.execution_hooks import (
    ExecutionHooks,
    ShieldedTaskSet,
)
from rayllm.backend.server.plugins.router_query_engine import (
    RouterQueryClient,
    StreamingErrorHandler,
)
from rayllm.backend.server.utils import (
    _replace_prefix,
    stream_model_responses,
)
from rayllm.common.models import DeletedModel, ModelData

logger = get_logger(__name__)


class ServeApplicationQueryClient(RouterQueryClient):
    """An implementation of the RouterQueryClient
    This RouterQueryClient uses HTTP to query the routes specified.
    It assumes that the relevant models are sitting at the specified routes.
    """

    def __init__(
        self,
        routes: Dict[str, str],
        engine_configurations: Dict[str, Args],
        hooks: Optional[ExecutionHooks] = None,
        port: int = 8000,
        metrics: Optional[Metrics] = None,
    ) -> None:
        # TODO (shrekris-anyscale): Remove self._routes once deployments can
        # stream results to other deployments. Use Serve handles instead.
        self._routes = routes
        # TODO: Remove this once it is possible to reconfigure models on the fly
        self._engine_configurations = engine_configurations
        self.hooks = hooks or ExecutionHooks()
        self.metrics = metrics or Metrics()
        self.port = port
        self.task_set = ShieldedTaskSet()
        self.metrics_wrapper = StreamingErrorHandler(
            self.hooks, self.metrics, self.task_set
        )

    async def stream(
        self,
        model: str,
        prompt: Prompt,
        request: Request,
        priority: QueuePriority,
    ) -> AsyncIterator[AviaryModelResponse]:
        route = self._get_model_path(model)
        url = f"http://localhost:{self.port}{route}/stream"
        json = {
            "prompt": prompt.dict(),
            "priority": int(priority),
        }
        raw_iterator = stream_model_responses(url, json=json)
        async for response in self.metrics_wrapper.handle_failure(
            model, request, prompt, raw_iterator
        ):
            yield response

    @property
    def all_models(self) -> Dict[str, ModelData]:
        model_ids = list(self._engine_configurations.keys())
        return {model_id: self._model(model_id) for model_id in model_ids}

    async def models(self):
        return self.all_models

    async def model(self, model_id: str):
        return self.all_models.get(model_id)

    async def delete_fine_tuned_model(self, model: str) -> DeletedModel:
        raise NotImplementedError

    def _get_model_path(self, model: str):
        model = _replace_prefix(model)
        route = self._routes.get(model)
        if route is None:
            raise OpenAIHTTPException(
                message=f"Invalid model '{model}'",
                status_code=status.HTTP_400_BAD_REQUEST,
                type="InvalidModel",
            )
        return route

    def _model(self, model: str):
        metadata = self._engine_configurations[model].short_metadata()
        return ModelData(
            id=model,
            object="model",
            owned_by="organization-owner",  # TODO
            permission=[],  # TODO
            rayllm_metadata=metadata,
        )
