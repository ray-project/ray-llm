import logging
import time
from typing import AsyncIterator

from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams as VLLMInternalSamplingParams

from rayllm.backend.llm.error_handling import ValidationError
from rayllm.backend.llm.generation import (
    FinishReason,
)
from rayllm.backend.llm.llm_node_initializer import LLMNodeInitializer
from rayllm.backend.llm.utils import BatchAviaryModelResponses
from rayllm.backend.llm.vllm.vllm_compatibility import AviaryAsyncLLMEngine
from rayllm.backend.llm.vllm.vllm_engine_stats import (
    ArgUsage,
    VLLMEngineStats,
    VLLMEngineStatTracker,
    usage_counters,
)
from rayllm.backend.llm.vllm.vllm_models import (
    VLLMApp,
    VLLMGenerationRequest,
    VLLMSamplingParams,
)
from rayllm.backend.observability.base import step
from rayllm.backend.observability.fn_call_metrics import (
    ClockUnit,
    MsClock,
)
from rayllm.backend.server.models import (
    AviaryModelResponse,
)

logger = logging.getLogger(__name__)


class VLLMEngine:
    _engine_cls = AviaryAsyncLLMEngine

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
        self.placement_config = llm_app.placement_config
        if not (self.placement_config.scaling_config.num_gpus_per_worker > 0):
            raise ValueError("The VLLM Engine Requires > 0 GPUs to run.")

        self.node_initializer = node_initializer or LLMNodeInitializer(
            local_node_tokenizer_only=True
        )
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
            self.engine = self._engine_cls.from_llm_app(
                self.llm_app,
                pg,
                runtime_env,
            )
            self.running = True

    async def generate(self, vllm_engine_request: VLLMGenerationRequest):
        response_stream = BatchAviaryModelResponses(self._generate(vllm_engine_request))
        async for response in response_stream.stream():
            yield response

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
        req_id = vllm_generation_request.request_id

        # Construct a results generator from VLLM
        results_generator: AsyncIterator[RequestOutput] = self.engine.generate(
            vllm_generation_request.prompt,
            self._parse_sampling_params(vllm_generation_request.sampling_params),
            req_id,
        )

        # Loop over the results
        num_text_returned = 0
        all_tokens_collected = 0
        clock = MsClock(unit=ClockUnit.s)
        output = None
        try:
            start = time.perf_counter()
            async for request_output in self._stats.auto_track(results_generator):
                # TODO(tchordia): handle more than one output
                assert (
                    len(request_output.outputs) == 1
                ), "Received more than 1 output from vllm, aborting"
                output = request_output.outputs[0]
                text_output = output.text[num_text_returned:]
                num_text_returned += len(text_output)
                num_input_tokens = len(request_output.prompt_token_ids)
                tokens_collected = len(output.token_ids) - all_tokens_collected
                all_tokens_collected += tokens_collected
                finish_reason = FinishReason.from_vllm_finish_reason(
                    output.finish_reason
                )
                yield AviaryModelResponse(
                    generated_text=text_output,
                    num_generated_tokens=tokens_collected,
                    num_generated_tokens_batch=tokens_collected,
                    num_input_tokens=num_input_tokens,
                    num_input_tokens_batch=num_input_tokens,
                    preprocessing_time=0,
                    generation_time=clock.reset_interval(),
                    finish_reason=finish_reason,
                )
            total_request_time = time.perf_counter() - start
            logger.info(
                f"Request {req_id} finished ({finish_reason}). "
                f"Total time: {total_request_time}s, "
                f"Input tokens: {num_input_tokens}, Generated tokens: {all_tokens_collected}, "
            )
        except Exception as e:
            logger.error(
                f"Failed while generating for requests ({req_id}): {repr(e)}",
                exc_info=e,
            )

        finally:
            # Ensure that we cancel on the engine once we have exited the streaming
            # phase
            self.engine._abort(req_id)

    async def check_health(self) -> bool:
        # TODO(tchordia): Add a healthcheck to the vllm engine
        return True

    def stats(self) -> VLLMEngineStats:
        return self._stats.to_stats()

    def shutdown(self, shutdown_pg: bool = True):
        raise NotImplementedError()

    def _collect_usage_metrics(self, sampling_params: VLLMSamplingParams) -> None:
        usage_counters[ArgUsage.BEST_OF].inc(
            1 if sampling_params.best_of is not None else 0
        )
        usage_counters[ArgUsage.PRESENCE_PENALTY].inc(
            1 if sampling_params.presence_penalty is not None else 0
        )
        usage_counters[ArgUsage.FREQUENCY_PENALTY].inc(
            1 if sampling_params.frequency_penalty is not None else 0
        )
        usage_counters[ArgUsage.PRESENCE_AND_FREQUENCY_PENALTY].inc(
            1
            if (
                sampling_params.presence_penalty is not None
                and sampling_params.frequency_penalty is not None
            )
            else 0
        )
        usage_counters[ArgUsage.TEMPERATURE].inc(
            1 if sampling_params.temperature is not None else 0
        )
        usage_counters[ArgUsage.TOP_P].inc(
            1 if sampling_params.top_p is not None else 0
        )
        usage_counters[ArgUsage.TOP_K].inc(
            1 if sampling_params.top_k is not None else 0
        )
        usage_counters[ArgUsage.STOP].inc(1 if sampling_params.stop is not None else 0)
        usage_counters[ArgUsage.MAX_TOKENS].inc(
            1 if sampling_params.max_tokens is not None else 0
        )
        usage_counters[ArgUsage.LOGPROBS].inc(
            1 if sampling_params.logprobs is not None else 0
        )

    def _parse_sampling_params(
        self, sampling_params: VLLMSamplingParams, **extra_fields
    ) -> VLLMInternalSamplingParams:
        try:
            if sampling_params.n != 1:
                raise ValueError("n>1 is not supported yet in aviary")
            self._collect_usage_metrics(sampling_params)
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
                **extra_fields,
            )
        except Exception as e:
            # Wrap the error in ValidationError so the status code
            # returned to the user is correct.
            raise ValidationError(str(e)) from e
