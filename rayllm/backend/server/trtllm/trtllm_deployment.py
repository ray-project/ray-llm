import logging

from ray import serve

from rayllm.backend.llm.trtllm.trtllm_engine import TRTLLMEngine
from rayllm.backend.llm.trtllm.trtllm_models import (
    TRTLLMApp,
    TRTLLMGenerationRequest,
    TRTLLMSamplingParams,
)
from rayllm.common.models import Prompt

logger = logging.getLogger(__name__)


@serve.deployment
class TRTLLMDeployment:
    _generation_request_cls = TRTLLMGenerationRequest

    def __init__(self, base_config: TRTLLMApp, generation_request_cls=None):
        self.engine = TRTLLMEngine(base_config)
        self.base_config = base_config
        self.generation_request_cls = (
            generation_request_cls or self._generation_request_cls
        )

    async def stream(
        self,
        request_id: str,
        prompt: Prompt,
        priority=None,
    ):
        sample_params = TRTLLMSamplingParams.parse_obj(prompt.parameters)
        logger.info(f"Received request {request_id}")

        prompt_text = (
            self.base_config.engine_config.generation.prompt_format.generate_prompt(
                prompt
            )
        )

        request = self.generation_request_cls(
            prompt=prompt_text,
            request_id=request_id,
            sampling_params=sample_params,
            stream=prompt.parameters.get("stream", True),
        )

        async for aviary_model_response in self.engine.generate(request):
            yield aviary_model_response
