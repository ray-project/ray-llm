import logging
import os

from ray import serve

from rayllm.backend.llm.vllm.vllm_engine import VLLMEngine
from rayllm.backend.llm.vllm.vllm_models import (
    VLLMApp,
    VLLMGenerationRequest,
    VLLMSamplingParams,
)
from rayllm.backend.server.models import (
    QueuePriority,
    SchedulingMetadata,
)
from rayllm.common.models import Prompt

logger = logging.getLogger(__name__)


class VLLMDeploymentImpl:
    _generation_request_cls = VLLMGenerationRequest
    _default_engine_cls = VLLMEngine

    async def __init__(
        self, base_config: VLLMApp, *, engine_cls=None, generation_request_cls=None
    ):
        self.base_config = base_config
        self.config_store = {}  # type: ignore

        engine_cls = engine_cls or self._default_engine_cls
        self._generation_request_cls = (
            generation_request_cls or self._generation_request_cls
        )

        self.engine = engine_cls(base_config)
        await self.engine.start()

    async def stream(
        self,
        request_id: str,
        prompt: Prompt,
        priority=QueuePriority.GENERATE_TEXT,
    ):
        """A thin wrapper around VLLMEngine.generate().
        1. Load the model to disk
        2. Format parameters correctly
        3. Forward request to VLLMEngine.generate()
        """

        prompt_text = (
            self.base_config.engine_config.generation.prompt_format.generate_prompt(
                prompt
            )
        )
        sampling_params = VLLMSamplingParams.merge_generation_params(
            prompt, self.base_config.engine_config.generation
        )

        logger.info(f"Received streaming request {request_id}")
        vllm_request = self._generation_request_cls(
            prompt=prompt_text,
            request_id=request_id,
            sampling_params=sampling_params,
            scheduling_metadata=SchedulingMetadata(
                request_id=request_id, priority=priority
            ),
        )
        async for aviary_model_response in self.engine.generate(vllm_request):
            yield aviary_model_response

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
class VLLMDeployment(VLLMDeploymentImpl):
    ...
