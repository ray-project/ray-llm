import asyncio
from random import randint

from vllm.sampling_params import SamplingParams as VLLMInternalSamplingParams

from aviary.backend.llm.error_handling import ValidationError
from aviary.backend.llm.generation import (
    FinishReason,
)
from aviary.backend.llm.vllm.vllm_engine_stats import (
    VLLMEngineStats,
    VLLMEngineStatTracker,
)
from aviary.backend.llm.vllm.vllm_models import (
    VLLMApp,
    VLLMGenerationRequest,
    VLLMSamplingParams,
)
from aviary.backend.llm.vllm.vllm_node_initializer import (
    VLLMNodeInitializer,
)
from aviary.backend.server.models import AviaryModelResponse


class MockVLLMEngine:
    def __init__(
        self, llm_app: VLLMApp, *, node_initializer: VLLMNodeInitializer = None
    ):
        """Create a VLLM Engine class

        Args:
            llm_app (VLLMApp): The Aviary configuration for this engine
        """
        assert isinstance(
            llm_app, VLLMApp
        ), f"Got invalid config {llm_app} of type {type(llm_app)}"
        self.llm_app = llm_app.copy(deep=True)
        self.engine_config = llm_app.engine_config

        self._stats = VLLMEngineStatTracker()

    async def start(self):
        """No-Op"""
        return

    @staticmethod
    async def async_range(count):
        for i in range(count):
            yield (i)
            await asyncio.sleep(0.0)

    async def generate(self, vllm_engine_request: VLLMGenerationRequest):
        sampling_params = self._parse_sampling_params(
            vllm_engine_request.sampling_params
        )
        max_tokens = sampling_params.max_tokens
        if not max_tokens:
            max_tokens = randint(1, 10)
        prompt = vllm_engine_request.prompt
        prompt_len = (
            len(prompt.split()) if isinstance(prompt, str) else len(prompt.prompt)
        )
        generation_time = 0.001
        async for i in self.async_range(max_tokens):
            if i == max_tokens - 1:
                finish_reason = FinishReason.STOP
            else:
                finish_reason = None
            aviary_model_response = AviaryModelResponse(
                generated_text=f"test_{i} ",
                num_input_tokens=prompt_len,
                num_input_tokens_batch=prompt_len,
                num_generated_tokens=1,
                preprocessing_time=0,
                generation_time=generation_time,
                finish_reason=finish_reason,
            )
            yield aviary_model_response
            await asyncio.sleep(generation_time)

    async def check_health(self) -> bool:
        return True

    def stats(self) -> VLLMEngineStats:
        return self._stats.to_stats()

    def shutdown(self, shutdown_pg: bool = True):
        raise NotImplementedError()

    def _parse_sampling_params(
        self, sampling_params: VLLMSamplingParams
    ) -> VLLMInternalSamplingParams:
        try:
            if sampling_params.n != 1:
                raise ValueError("n>1 is not supported yet in aviary")
            return VLLMInternalSamplingParams(
                n=1,
                best_of=sampling_params.best_of,
                presence_penalty=sampling_params.presence_penalty
                if sampling_params.presence_penalty is not None
                else 0.0,
                frequency_penalty=sampling_params.frequency_penalty
                if sampling_params.frequency_penalty is not None
                else 0.0,
                temperature=sampling_params.temperature
                if sampling_params.temperature is not None
                else 1.0,
                top_p=sampling_params.top_p
                if sampling_params.top_p is not None
                else 1.0,
                top_k=sampling_params.top_k
                if sampling_params.top_k is not None
                else -1,
                use_beam_search=False,
                stop=sampling_params.stop,
                ignore_eos=False,
                # vLLM will cancel internally if input+output>max_tokens
                max_tokens=sampling_params.max_tokens
                or self.engine_config.max_total_tokens,
                logprobs=sampling_params.logprobs,
            )
        except Exception as e:
            # Wrap the error in ValidationError so the status code
            # returned to the user is correct.
            raise ValidationError(str(e)) from e
