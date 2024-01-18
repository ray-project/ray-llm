import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type, Union

import ray
from ray.util.placement_group import PlacementGroup
from transformers.dynamic_module_utils import init_hf_modules
from vllm.config import CacheConfig as VllmCacheConfig
from vllm.config import ModelConfig as VllmModelConfig
from vllm.config import ParallelConfig as VllmParallelConfig
from vllm.config import SchedulerConfig as VllmSchedulerConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine, AsyncStream, _AsyncLLMEngine

from rayllm.backend.llm.error_handling import InputTooLong
from rayllm.backend.llm.vllm.metrics.vllm_compatibility import (
    engine_metrics,
    engine_record_stats_gauges,
    running_requests_gauge,
    swapped_requests_gauge,
    waiting_requests_gauge,
)
from rayllm.backend.llm.vllm.vllm_models import VLLMApp

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams


# This has to be ran at module level to fix serialization issues
# with remote HF Transformers code.
init_hf_modules()

logger = logging.getLogger(__name__)

VllmConfigs = Tuple[
    VllmCacheConfig, VllmModelConfig, VllmParallelConfig, VllmSchedulerConfig
]


class AviaryLLMEngine(_AsyncLLMEngine):
    def __init__(self, *args, runtime_env: dict, **kwargs):
        self.runtime_env = runtime_env
        super().__init__(*args, **kwargs)

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        ray_remote_kwargs.setdefault("runtime_env", self.runtime_env)
        return super()._init_workers_ray(placement_group, **ray_remote_kwargs)

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
        if hasattr(self.model_config, "get_max_model_len"):
            max_input_length = self.model_config.get_max_model_len()
        else:
            max_input_length = self.model_config.max_model_len
        if num_input_tokens > max_input_length:
            logger.info(
                f"Task {request_id} is over the max input length ({num_input_tokens}/{max_input_length})."
            )
            raise InputTooLong(num_input_tokens, max_input_length).exception
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
            else:
                raise ValueError(f"Unknown stat {name}")
        return last_stats


def _get_vllm_engine_config(vllm_app) -> Tuple[AsyncEngineArgs, VllmConfigs]:
    # Generate engine arguements and engine configs

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
    configs = async_engine_args.create_engine_configs()
    return async_engine_args, configs


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
        if arrival_time is None:
            arrival_time = time.time()
        prompt_token_ids = await self.engine._encode_request_async(
            request_id, prompt, prompt_token_ids
        )
        return await super().add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            arrival_time=arrival_time,
            **kwargs,
        )

    @classmethod
    def from_llm_app(
        cls,
        vllm_app: VLLMApp,
        placement_group: PlacementGroup,
        runtime_env: dict,
    ) -> "AviaryAsyncLLMEngine":
        """Creates an async LLM engine from the engine arguments."""

        scaling_config = vllm_app.get_scaling_options(placement_group)
        # When using gpu special type, vllm does a type check that requires
        # torch to have access to CUDA devices. We use a remote task
        # with `num_gpus` set here, so the type check happens in an environment
        # with `CUDA_VISIBLE_DEVICES` set.
        engine_args, engine_configs = ray.get(
            ray.remote(_get_vllm_engine_config)
            .options(**scaling_config)
            .remote(vllm_app)
        )

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
