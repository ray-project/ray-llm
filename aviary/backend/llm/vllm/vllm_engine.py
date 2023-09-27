import logging
import time
from typing import AsyncIterator

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams as VLLMInternalSamplingParams

from aviary.backend.llm.error_handling import ValidationError
from aviary.backend.llm.generation import (
    FinishReason,
)
from aviary.backend.llm.vllm.util import BatchAviaryModelResponses
from aviary.backend.llm.vllm.vllm_compatibility import AviaryAsyncLLMEngine
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
from aviary.backend.observability.base import step
from aviary.backend.observability.fn_call_metrics import (
    ClockUnit,
    MsClock,
)
from aviary.backend.server.models import (
    AviaryModelResponse,
)

logger = logging.getLogger(__name__)


class VLLMEngine:
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
        self.placement_config = llm_app.placement_config
        if not (self.placement_config.scaling_config.num_gpus_per_worker > 0):
            raise ValueError("The VLLM Engine Requires > 0 GPUs to run.")

        self.node_initializer = node_initializer or VLLMNodeInitializer()
        self._stats = VLLMEngineStatTracker()
        self.running = False

    async def start(self):
        """Start the VLLM Engine

        If the engine is already running, do nothing.

        1. First initialize the node by downloading the model from s3
        2. Next instantiate the VLLM engine by constructing a config from the LLMApp.

        """
        if self.running:
            # The engine is already running!
            logger.info("Skipping engine restart because the engine is already running")
            return

        # Get the scaling options
        with step("Starting vllm engine", request_id="node_initialize"):
            pg, runtime_env = await self.node_initializer.initialize_node(self.llm_app)

            # Make sure VLLM uses our placement group & runtime env
            self.engine = AviaryAsyncLLMEngine.from_llm_app(
                self.llm_app,
                pg,
                runtime_env,
            )
            self.running = True

    async def generate(self, vllm_engine_request: VLLMGenerationRequest):
        response_stream = BatchAviaryModelResponses(self._generate(vllm_engine_request))
        async for response in response_stream.stream():
            yield response

    async def _get_results_generator(
        self, vllm_generation_request: VLLMGenerationRequest
    ):
        results_generator: AsyncIterator[RequestOutput] = self.engine.generate(
            vllm_generation_request.prompt,
            self._parse_sampling_params(vllm_generation_request.sampling_params),
            vllm_generation_request.request_id,
        )
        return results_generator

    async def _generate(
        self, vllm_generation_request: VLLMGenerationRequest
    ) -> AsyncIterator[AviaryModelResponse]:
        """Generate an AviaryModelResponse stream

        The VLLM generation request will be passed into VLLM, and the resulting output
        will be wrapped in an AviaryModelResponse and yielded back to the user.

        Error handling:

        We schedule a finalizer that will abort the request on the engine.

        If an exception is raised in this function or vllm, the finalizer guarantees that the request is aborted.
        If an exception is raised in the caller, when this generator is gced, it will run the finalizer and abort the request.

        This should also handle the case where the caller is cancelled (raises asyncio.CancelledError)
        """
        # Construct a results generator from VLLM
        results_generator = await self._get_results_generator(vllm_generation_request)

        # Loop over the results
        num_text_returned = 0
        clock = MsClock(unit=ClockUnit.s)
        output = None
        try:
            start = time.perf_counter()
            tokens_collected = 0
            async for request_output in self._stats.auto_track(results_generator):
                # TODO(tchordia): handle more than one output
                assert (
                    len(request_output.outputs) == 1
                ), "Received more than 1 output from vllm, aborting"
                output = request_output.outputs[0]
                text_output = output.text[num_text_returned:]
                num_text_returned += len(text_output)
                num_input_tokens = len(request_output.prompt_token_ids)
                tokens_collected += 1
                finish_reason = FinishReason.from_vllm_finish_reason(
                    output.finish_reason
                )
                yield AviaryModelResponse(
                    generated_text=text_output,
                    num_generated_tokens=1,
                    num_generated_tokens_batch=1,
                    num_input_tokens=num_input_tokens,
                    num_input_tokens_batch=num_input_tokens,
                    preprocessing_time=0,
                    generation_time=clock.reset_interval(),
                    finish_reason=finish_reason,
                )
            total_request_time = time.perf_counter() - start
            logger.info(
                f"Request {vllm_generation_request.request_id} finished ({finish_reason}). "
                f"Total time: {(total_request_time)}s, "
                f"Input tokens: {num_input_tokens}, Generated tokens: {tokens_collected}, "
            )
        finally:
            # Ensure that we cancel on the engine once we have exited the streaming
            # phase
            self.engine._abort(vllm_generation_request.request_id)

    def check_health(self) -> bool:
        # TODO(tchordia): Add a healthcheck to the vllm engine
        raise NotImplementedError()

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
