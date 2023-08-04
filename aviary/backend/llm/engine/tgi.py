import asyncio
import gc
import os
import time
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
)

import ray
import ray.exceptions
import ray.util
from ray.air import ScalingConfig
from ray.air.util.torch_dist import TorchDistributedWorker
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer

from aviary.backend.llm.continuous.error_handling import InputTooLong
from aviary.backend.llm.continuous.policy import QuotaBasedTaskSelectionPolicy
from aviary.backend.llm.continuous.scheduler import (
    InferenceScheduler,
)
from aviary.backend.llm.continuous.tokenizer import (
    CachingTokenizer,
    TransformersTokenizer,
)
from aviary.backend.llm.continuous.tokenstream import FinishReason
from aviary.backend.llm.continuous.types import InferenceTask, Request, TGIParams
from aviary.backend.llm.utils import (
    _init_torch_distributed_env_vars_only,
    init_torch_dist_process_group_async,
    initialize_node,
)
from aviary.backend.logger import get_logger
from aviary.backend.server.models import (
    AviaryModelResponse,
    TextGenerationInferenceEngineConfig,
)
from aviary.backend.server.utils import QueuePriority
from aviary.common.models import Prompt
from aviary.conf import ENV_VARS_TO_PROPAGATE

from ..utils import (
    get_model_location_on_disk,
    merge_dicts,
)

try:
    from aviary.backend.llm.continuous.tgi.tgi_worker import (
        TGIInferenceWorker,
        TGIInferenceWorkerGroup,
    )
except ImportError as e:
    TGIInferenceWorkerGroup = e

    class TGIInferenceWorker:
        pass


logger = get_logger(__name__)

TOTAL_BATCH_TOKENS_MULTIPLIER = 0.99


# We need to inherit from TorchDistributedWorker for compatibility with
# ray.air.utils.init_torch_dist_process_group
class AviaryTGIInferenceWorker(TGIInferenceWorker, TorchDistributedWorker):
    """A InferenceWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple InferenceWorker of the same WorkerGroup will form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.

    Args:
        engine_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
    """

    def __init__(
        self, engine_config: TextGenerationInferenceEngineConfig, world_size: int
    ):
        self.engine_config = engine_config
        self.world_size = world_size
        self.local_rank = None
        self.current_device = None
        self.gpu_memory_fraction = 1.0

        model_id = self.engine_config.actual_hf_model_id
        model_id_or_location = get_model_location_on_disk(
            self.engine_config.actual_hf_model_id
        )
        if model_id != model_id_or_location:
            safetensor_files = list(Path(model_id_or_location).glob("*.safetensors"))
            if safetensor_files:
                model_id = model_id_or_location

        super().__init__(
            model_id=model_id,
            **self.engine_config.get_initialization_kwargs(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.engine_config.model_id}"

    def ping(self) -> bool:
        """Ping the worker."""
        return True


class TextGenerationInferenceEngine:
    _prediction_worker_cls: Type[AviaryTGIInferenceWorker] = AviaryTGIInferenceWorker

    def __init__(
        self,
        engine_config: Optional[TextGenerationInferenceEngineConfig],
    ) -> None:
        if isinstance(TGIInferenceWorkerGroup, Exception):
            raise RuntimeError(
                "TextGenerationInferenceEngine requires Yard1/text-generation-inference to be installed. "
                "The best way to ensure that the environment is set up correctly is to use the Aviary "
                "Docker image - refer to https://github.com/ray-project/ray/ for instructions."
            ) from TGIInferenceWorkerGroup
        self.engine_config = engine_config
        self.base_worker_group = None
        self.new_worker_group = None
        self._base_worker_group_lock = asyncio.Lock()
        self._new_worker_group_lock = asyncio.Lock()
        self.scheduler = None
        self.tokenizer = None

    def is_initialized(self) -> bool:
        return bool(self.base_worker_group)

    async def rollover(self, scaling_config: ScalingConfig, pg_timeout_s: float = 600):
        """Roll over to a new worker group.

        The new worker group is created asynchronously and the old worker group
        is replaced with the new worker group once it is ready.

        Args:
            scaling_config (ScalingConfig): Scaling configuration for the new worker group.
            pg_timeout_s (float, optional): Timeout for the new worker group to be ready. Defaults to 600.
        """
        if self._new_worker_group_lock.locked():
            logger.info("Rollover already in progress")
            return
        async with self._new_worker_group_lock:
            logger.info(f"Initializing new worker group {scaling_config}")
            self.new_worker_group = await self._create_worker_group(
                scaling_config, pg_timeout_s=pg_timeout_s
            )
            # async with self._base_worker_group_lock:
            logger.info(f"Rolling over to new worker group {self.new_worker_group}")
            self.base_worker_group = self.new_worker_group
            self.new_worker_group = None
            gc.collect()

    async def _initialize_torch_dist_process_group(
        self, worker_group: List[ray.ObjectRef], **kwargs
    ) -> List[int]:
        return await init_torch_dist_process_group_async(
            worker_group, init_function=_init_torch_distributed_env_vars_only, **kwargs
        )

    async def _start_prediction_workers(
        self, scaling_config: ScalingConfig, remote_prediction_worker_cls: type
    ):
        # Create the prediction workers.
        logger.info("Creating prediction workers...")
        worker_group = [
            remote_prediction_worker_cls.remote(
                self.engine_config, scaling_config.num_workers
            )
            for i in range(scaling_config.num_workers)
        ]

        logger.info("Initializing torch_dist process group on workers...")
        # Initialize torch distributed process group for the workers.
        local_ranks = await self._initialize_torch_dist_process_group(
            worker_group,
            backend="nccl" if scaling_config.use_gpu else "gloo",
        )

        # Initialize model on each worker.
        logger.info("Initializing model on workers...")
        await asyncio.gather(
            *[
                worker.init_model.remote(
                    local_rank,
                    num_cpus_per_worker=scaling_config.num_cpus_per_worker,
                    num_gpus_per_worker=scaling_config.num_gpus_per_worker,
                )
                for worker, local_rank in zip(worker_group, local_ranks)
            ]
        )

        logger.info("Warming up model on workers...")

        can_infer_max_batch_total_tokens = (
            await asyncio.gather(
                worker_group[0].can_infer_max_batch_total_tokens.remote()
            )
        )[0]
        if can_infer_max_batch_total_tokens:
            max_batch_total_tokens = None
        else:
            max_batch_total_tokens = self.task_selection_policy.max_batch_total_tokens
            if not max_batch_total_tokens:
                raise ValueError(
                    f"Model {self.engine_config.model_id} cannot automatically infer max_batch_total_tokens. "
                    "Make sure to set engine_config.scheduler.policy.max_batch_total_tokens in the model "
                    "configuration yaml."
                )

        max_supported_total_tokens = await asyncio.gather(
            *[
                worker.warmup.remote(
                    max_batch_prefill_tokens=self.task_selection_policy.max_batch_prefill_tokens,
                    max_input_length=self.task_selection_policy.max_input_length,
                    max_batch_total_tokens=max_batch_total_tokens,
                )
                for worker in worker_group
            ]
        )

        max_supported_total_tokens = min(max_supported_total_tokens)

        if can_infer_max_batch_total_tokens and max_supported_total_tokens:
            self.task_selection_policy.max_batch_total_tokens = int(
                max_supported_total_tokens * TOTAL_BATCH_TOKENS_MULTIPLIER
            )

            # Warmup again with max_supported_total_tokens to ensure constant environment across workers
            max_supported_total_tokens = await asyncio.gather(
                *[
                    worker.warmup.remote(
                        max_batch_prefill_tokens=self.task_selection_policy.max_batch_prefill_tokens,
                        max_input_length=self.task_selection_policy.max_input_length,
                        max_batch_total_tokens=self.task_selection_policy.max_batch_total_tokens,
                    )
                    for worker in worker_group
                ]
            )
            max_supported_total_tokens = min(max_supported_total_tokens)

        if max_supported_total_tokens:
            self.task_selection_policy.max_batch_total_tokens = (
                max_supported_total_tokens
            )

        assert worker_group
        return worker_group

    def _prepare_worker_runtime_env(self) -> dict:
        runtime_env = self.engine_config.runtime_env or {}
        runtime_env.setdefault("env_vars", {})
        for env_var in ENV_VARS_TO_PROPAGATE:
            if env_var in os.environ:
                runtime_env["env_vars"][env_var] = os.getenv(env_var)
        return runtime_env

    def _get_initialize_node_fn(self) -> Callable:
        return initialize_node

    async def _create_worker_group(
        self,
        scaling_config: ScalingConfig,
        pg_timeout_s: float = 600,
    ) -> List[ray.ObjectRef]:
        assert self.engine_config

        self.task_queue = asyncio.Queue()
        self.task_selection_policy = QuotaBasedTaskSelectionPolicy(
            **self.engine_config.scheduler.policy.dict(exclude={"type"})
        )

        assert self.engine_config
        llm_config = self.engine_config

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        scaling_options = dict(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )
        runtime_env = self._prepare_worker_runtime_env()
        remote_prediction_worker_cls = ray.remote(self._prediction_worker_cls).options(
            **scaling_options, runtime_env=runtime_env
        )

        initialize_node = self._get_initialize_node_fn()
        initialize_node_remote = ray.remote(initialize_node)
        initialize_node_remote_pg = initialize_node_remote.options(
            **scaling_options, runtime_env=runtime_env
        )

        logger.info("Waiting for placement group to be ready...")
        # This will raise a timeout error.
        try:
            await asyncio.wait_for(self.pg.ready(), timeout=pg_timeout_s)
        except asyncio.TimeoutError as e:
            raise RuntimeError(
                f"Placement group {self.pg} did not become ready within {pg_timeout_s} seconds. "
                "This means that the cluster doesn't have the required resources to start the worker group. "
                "Please check the autoscaler logs for more information.\n"
                "This can also be caused by the model workers requiring resources that are not present in the "
                "cluster (eg. `accelerator_type_a10`). Either remove them from the model configuration yaml "
                "or add them to the cluster."
            ) from e

        logger.info("Starting initialize_node tasks...")
        await asyncio.gather(
            *[
                initialize_node_remote_pg.remote(
                    llm_config.actual_hf_model_id,
                    llm_config.s3_mirror_config,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # Download just the tokenizer for the engine
        path = initialize_node(
            llm_config.actual_hf_model_id,
            llm_config.s3_mirror_config,
            tokenizer_only=True,
        )
        if path:
            llm_config.hf_model_id = path

        # Download the tokenizer
        _ = AutoTokenizer.from_pretrained(llm_config.actual_hf_model_id)

        worker_group = await self._start_prediction_workers(
            scaling_config=scaling_config,
            remote_prediction_worker_cls=remote_prediction_worker_cls,
        )

        self.tokenizer = CachingTokenizer(
            TransformersTokenizer(ray.get(worker_group[0].get_tokenizer.remote())),
            capacity=1024,
        )
        self.inference_task_cls = ray.get(
            worker_group[0].get_inference_task_cls.remote()
        )

        self.scheduler = InferenceScheduler(
            inference_worker=TGIInferenceWorkerGroup(worker_group=worker_group),
            task_selection_policy=self.task_selection_policy,
            task_queue=self.task_queue,
        )

        return worker_group

    def process_request(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        sampling_params: Dict[str, Any],
    ) -> InferenceTask:
        num_input_tokens = self.tokenizer.get_input_length(prompt)
        if num_input_tokens > self.task_selection_policy.max_input_length:
            logger.info("Task is over the max input length.")
            InputTooLong(
                num_input_tokens, self.task_selection_policy.max_input_length
            ).raise_exception()

        if "stopping_sequences" in sampling_params:
            sampling_params["stop_sequences"] = sampling_params.pop(
                "stopping_sequences"
            )
        max_new_tokens = int(
            min(
                max_new_tokens or float("inf"),
                self.task_selection_policy.max_total_tokens - num_input_tokens,
            )
        )
        task = self.inference_task_cls(
            Request(
                inputs=prompt,
                input_tokens=num_input_tokens,
                truncate=self.task_selection_policy.max_input_length,
                max_new_tokens=max_new_tokens,
                params=TGIParams(**sampling_params),
            )
        )
        self.scheduler.add_task(task)
        return task

    def validate_prompt(self, prompt: Prompt) -> None:
        # No validation here - instead, it happens inside stream_async.
        pass

    async def stream_async(
        self, prompt: Prompt, priority: QueuePriority
    ) -> AsyncGenerator[AviaryModelResponse, None]:
        """Generate text for a list of prompts.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.

        Returns:
            A list of generated texts.
        """
        prompt_text = self.engine_config.generation.prompt_format.generate_prompt(
            prompt
        )

        stopping_sequences = self.engine_config.generation.stopping_sequences or []
        stopping_sequences += prompt.stopping_sequences or []
        generate_kwargs = merge_dicts(
            {k: v for k, v in prompt.parameters.items() if v is not None} or {},
            self.engine_config.generation.generate_kwargs,
        )
        max_new_tokens = generate_kwargs.pop("max_new_tokens", None)
        result = self.process_request(
            prompt_text,
            max_new_tokens=max_new_tokens,
            sampling_params={
                **generate_kwargs,
                "stopping_sequences": stopping_sequences,
            },
        )
        token_stream = result.output_stream
        request_id = result.id
        try:
            start_time = time.monotonic()
            async for item in token_stream:
                # TODO maybe make the Scheduler/TokenStream return a Response directly
                yield AviaryModelResponse(
                    generated_text=item,
                    num_generated_tokens=1,
                    num_generated_tokens_batch=1,
                    num_input_tokens=result.input_length,
                    num_input_tokens_batch=result.input_length,
                    finish_reason=token_stream.finish_reason,
                    preprocessing_time=0,
                    generation_time=time.monotonic() - start_time,
                )
                start_time = time.monotonic()
            if token_stream.error_reason:
                token_stream.error_reason.raise_exception()
            yield AviaryModelResponse(
                finish_reason=token_stream.finish_reason,
            )
        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for {request_id}")
            token_stream.end(FinishReason.CANCELLED)
            raise
        except Exception:
            token_stream.end(FinishReason.ERROR)
            raise

    def check_health(self) -> None:
        if self._new_worker_group_lock.locked():
            logger.info("Rollover in progress, skipping health check")
            return
        if self.pg and self.base_worker_group:
            dead_actors = []
            for actor in self.base_worker_group:
                actor_state = ray.state.actors(actor._ray_actor_id.hex())
                if actor_state["State"] == "DEAD":
                    dead_actors.append(actor)
            if dead_actors:
                raise RuntimeError(
                    f"At least one prediction worker is dead. Dead workers: {dead_actors}. "
                    "Reinitializing worker group."
                )

        if self.scheduler:
            self.scheduler.check_health()
