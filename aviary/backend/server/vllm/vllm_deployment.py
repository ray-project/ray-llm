import logging
import os

from ray import serve

from aviary.backend.llm.vllm.vllm_engine import VLLMEngine
from aviary.backend.llm.vllm.vllm_models import (
    VLLMApp,
    VLLMGenerationRequest,
    VLLMSamplingParams,
)
from aviary.backend.server.models import (
    QueuePriority,
    SchedulingMetadata,
)
from aviary.common.models import Prompt

logger = logging.getLogger(__name__)


class VLLMDeploymentImpl:
    async def __init__(self, base_config: VLLMApp, *, _engine=None):
        self.base_config = base_config
        self.config_store = {}  # type: ignore

        self.engine = _engine or VLLMEngine(base_config)
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
        vllm_request = VLLMGenerationRequest(
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
        # TODO(tchordia): we need to actually check health here
        return True


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
)
class VLLMDeployment(VLLMDeploymentImpl):
    ...
