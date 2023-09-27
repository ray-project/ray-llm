import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional, Type, Union

from ray.util.placement_group import PlacementGroup
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream, _AsyncLLMEngine

from aviary.backend.llm.error_handling import InputTooLong
from aviary.backend.llm.vllm.metrics.vllm_compatibility import (
    engine_metrics,
    engine_record_stats_gauges,
    running_requests_gauge,
    swapped_requests_gauge,
    waiting_requests_gauge,
)
from aviary.backend.llm.vllm.vllm_models import VLLMApp

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams
    from vllm.worker.worker import Worker

logger = logging.getLogger(__name__)


class AviaryLLMEngine(_AsyncLLMEngine):
    def __init__(self, *args, runtime_env: dict, **kwargs):
        self.runtime_env = runtime_env
        super().__init__(*args, **kwargs)

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        ray_remote_kwargs.setdefault("runtime_env", self.runtime_env)
        return super()._init_workers_ray(placement_group, **ray_remote_kwargs)

    def _get_worker_cls(self) -> Type["Worker"]:
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from aviary.backend.llm.vllm.vllm_worker import (
            InstrumentedWorker,
        )

        return InstrumentedWorker

    @engine_metrics.wrap
    async def add_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: "SamplingParams",
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        **kwargs,
    ) -> Optional[Exception]:
        """Returns None if request was added sucessfully, otherwise returns an Exception object."""
        del kwargs
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self._encode_request_async(
            request_id, prompt, prompt_token_ids
        )

        if isinstance(prompt_token_ids, Exception):
            # This should NOT be raise.
            # The background loop will put it in a relevant AsyncStream,
            # which will raise it in the generate task associated
            # with the request. Raising it here would kill
            # the scheduler loop.
            return prompt_token_ids

        return self.add_request(
            request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=arrival_time,
        )

    async def _encode_request_async(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: Optional[List[int]] = None,
        tokenizer: Any = None,
    ) -> Union[List[int], Exception]:
        if prompt_token_ids is None:
            assert prompt is not None
            if tokenizer is None:
                tokenizer = self.tokenizer
            prompt_token_ids = tokenizer.encode(prompt)
        num_input_tokens = len(prompt_token_ids)
        max_input_length = self.model_config.get_max_model_len()
        if num_input_tokens > max_input_length:
            logger.info(
                f"Task {request_id} is over the max input length ({num_input_tokens}/{max_input_length})."
            )
            # Return an exception object so it can be raised in the right asyncio
            # task
            return InputTooLong(num_input_tokens, max_input_length).exception
        return prompt_token_ids

    @engine_metrics.wrap
    def abort_request(self, *args, **kwargs):
        return super().abort_request(*args, **kwargs)

    @engine_metrics.wrap
    def _schedule(self, *args, **kwargs):
        seq_group_metadata_list, scheduler_outputs, early_return = super()._schedule(
            *args, **kwargs
        )
        running_requests_gauge.set(len(self.scheduler.running))
        waiting_requests_gauge.set(len(self.scheduler.waiting))
        swapped_requests_gauge.set(len(self.scheduler.swapped))
        return seq_group_metadata_list, scheduler_outputs, early_return

    @engine_metrics.wrap
    def _process_model_outputs(self, *args, **kwargs):
        return super()._process_model_outputs(*args, **kwargs)

    @engine_metrics.wrap
    async def step_async(self, *args, **kwargs):
        return await super().step_async(*args, **kwargs)

    @engine_metrics.wrap
    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def _record_system_stats(self, *args, **kwargs):
        _, last_stats = super()._record_system_stats(*args, **kwargs)
        for name, stat in last_stats.items():
            if name in engine_record_stats_gauges:
                engine_record_stats_gauges[name].set(stat)
        return last_stats


class AviaryAsyncLLMEngine(AsyncLLMEngine):
    _engine_class: Type[_AsyncLLMEngine] = AviaryLLMEngine

    async def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: "SamplingParams",
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
        **kwargs,
    ) -> AsyncStream:
        del kwargs
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.engine._encode_request_async(
            request_id, prompt, prompt_token_ids
        )
        if isinstance(prompt_token_ids, Exception):
            raise prompt_token_ids
        return await super().add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        placement_group: PlacementGroup,
        runtime_env: dict,
    ) -> "AsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        # Create the async LLM engine.
        engine = cls(
            engine_args.worker_use_ray,
            engine_args.engine_use_ray,
            *engine_configs,
            None,
            placement_group,
            runtime_env=runtime_env,
            log_requests=not engine_args.disable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            max_log_len=engine_args.max_log_len,
            start_engine_loop=True,
        )
        return engine

    @classmethod
    def from_llm_app(
        cls,
        vllm_app: VLLMApp,
        placement_group: PlacementGroup,
        runtime_env: dict,
    ) -> "AviaryAsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""
        async_engine_args = AsyncEngineArgs(
            # This is the local path on disk, or the hf model id
            # If it is the hf_model_id, vllm automatically downloads the correct model.
            **dict(
                model=vllm_app.engine_config.actual_hf_model_id,
                worker_use_ray=True,
                engine_use_ray=False,
                tensor_parallel_size=vllm_app.placement_config.world_size,
                max_model_len=vllm_app.engine_config.max_total_tokens,
                disable_log_stats=False,
                max_log_len=64,
                **vllm_app.engine_config.get_initialization_kwargs(),
            )
        )
        return cls.from_engine_args(async_engine_args, placement_group, runtime_env)
