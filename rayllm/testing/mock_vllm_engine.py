import asyncio
from random import randint

from vllm.sampling_params import SamplingParams as VLLMInternalSamplingParams

from rayllm.backend.llm.error_handling import ValidationError
from rayllm.backend.llm.generation import (
    FinishReason,
)
from rayllm.backend.llm.llm_node_initializer import (
    LLMNodeInitializer,
)
from rayllm.backend.llm.vllm.vllm_engine_stats import (
    VLLMEngineStats,
    VLLMEngineStatTracker,
)
from rayllm.backend.llm.vllm.vllm_models import (
    VLLMApp,
    VLLMGenerationRequest,
    VLLMSamplingParams,
)
from rayllm.backend.server.models import AviaryModelResponse
from rayllm.common.models import LogProb, LogProbs


class MockVLLMEngine:
    def __init__(
        self, llm_app: VLLMApp, *, node_initializer: LLMNodeInitializer = None
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
                logprobs=self.get_logprobs(i, vllm_engine_request, sampling_params),
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
            if sampling_params.logprobs:
                if sampling_params.top_logprobs:
                    if not (0 <= sampling_params.top_logprobs <= 5):
                        raise ValueError("top_logprobs must be between 0 and 5")
                    log_probs = sampling_params.top_logprobs
                else:
                    log_probs = 1
            else:
                if sampling_params.top_logprobs:
                    raise ValueError(
                        "if top_logprobs is specified, logprobs must be set to `True`"
                    )
                log_probs = None

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
                logprobs=log_probs,
            )
        except Exception as e:
            # Wrap the error in ValidationError so the status code
            # returned to the user is correct.
            raise ValidationError(str(e)) from e

    def get_logprobs(
        self,
        i: int,
        vllm_engine_request: VLLMGenerationRequest,
        sampling_params: VLLMSamplingParams,
    ):
        """Helper function for generating AviaryModelResponse logprobs"""
        num_logprobs = sampling_params.logprobs
        top_logprobs = vllm_engine_request.sampling_params.top_logprobs
        if num_logprobs:
            log_probs = [
                LogProbs.create(
                    logprobs=[
                        LogProb(
                            logprob=0.0,
                            token=(
                                f"test_{i} " if idx == 0 else f"candidate_token_{idx}"
                            ),
                            bytes=[],
                        )
                        for idx in range(num_logprobs)
                    ],
                    top_logprobs=top_logprobs,
                )
            ]
        else:
            log_probs = None

        return log_probs
