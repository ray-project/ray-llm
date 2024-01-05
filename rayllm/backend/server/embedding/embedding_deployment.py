import logging
import os
from typing import List, Optional, Union

from ray import serve

from rayllm.backend.llm.embedding.embedding_engine import EmbeddingEngine
from rayllm.backend.llm.embedding.embedding_models import EmbeddingApp
from rayllm.backend.server.models import (
    AviaryModelResponse,
    GenerationRequest,
    QueuePriority,
    SchedulingMetadata,
)
from rayllm.backend.server.utils import get_response_for_error
from rayllm.common.models import Prompt

logger = logging.getLogger(__name__)


class EmbeddingDeploymentImpl:
    _generation_request_cls = GenerationRequest
    _default_engine_cls = EmbeddingEngine

    async def __init__(
        self, base_config: EmbeddingApp, *, engine_cls=None, generation_request_cls=None
    ):
        self.base_config = base_config
        self.config_store = {}  # type: ignore

        engine_cls = engine_cls or self._default_engine_cls
        self._generation_request_cls = (
            generation_request_cls or self._generation_request_cls
        )

        self.stream.set_max_batch_size(base_config.engine_config.max_batch_size)
        self.stream.set_batch_wait_timeout_s(
            base_config.engine_config.batch_wait_timeout_s
        )

        self.engine = engine_cls(base_config)
        await self.engine.start()

    def _parse_response(
        self, response: Union[AviaryModelResponse, Exception], request_id: str
    ):
        if isinstance(response, Exception):
            return get_response_for_error(response, request_id=request_id)
        return response

    @serve.batch(max_batch_size=1, batch_wait_timeout_s=0.1)
    async def stream(
        self,
        request_ids: List[str],
        prompts: List[Prompt],
        priorities: Optional[List[QueuePriority]] = None,
    ):
        """A thin wrapper around EmbeddingEngine.generate().
        1. Load the model to disk
        2. Format parameters correctly
        3. Forward request to EmbeddingEngine.generate()
        """
        if not priorities:
            priorities = [QueuePriority.GENERATE_TEXT for _ in request_ids]

        prompt_texts = [prompt.prompt for prompt in prompts]

        logger.info(
            f"Received streaming requests ({len(request_ids)}) {','.join(request_ids)}"
        )
        embedding_request = self._generation_request_cls(
            prompt=prompt_texts,
            request_id=request_ids,
            sampling_params=None,
            scheduling_metadata=SchedulingMetadata(
                request_id=request_ids, priority=priorities
            ),
        )

        async for batched_aviary_model_response in self.engine.generate(
            embedding_request
        ):
            logger.info(
                f"Finished generating for streaming requests ({len(request_ids)}) {','.join(request_ids)}"
            )
            yield [
                self._parse_response(response, request_id)
                for response, request_id in zip(
                    batched_aviary_model_response, request_ids
                )
            ]

    async def check_health(self):
        return await self.engine.check_health()


@serve.deployment(
    # TODO make this configurable in aviary run
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 10,
        "target_num_ongoing_requests_per_replica": int(
            os.environ.get("AVIARY_ROUTER_TARGET_NUM_ONGOING_REQUESTS_PER_REPLICA", 10)
        ),
    },
    max_concurrent_queries=20,  # Maximum backlog for a single replica
    health_check_period_s=30,
    health_check_timeout_s=30,
)
class EmbeddingDeployment(EmbeddingDeploymentImpl):
    ...
