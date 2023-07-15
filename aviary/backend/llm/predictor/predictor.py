import asyncio
import gc
import os
import traceback
from typing import Iterator, List, Optional, Type

import ray
import ray.util
import torch
import torch.backends.cuda
from ray.air import ScalingConfig
from ray.air.util.torch_dist import TorchDistributedWorker
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from aviary.backend.llm.exceptions import PromptTooLongError
from aviary.backend.llm.initializers import get_initializer_cls_by_name
from aviary.backend.llm.pipelines import get_pipeline_cls_by_name
from aviary.backend.llm.pipelines._base import (
    AsyncStreamingPipeline,
    StreamingPipeline,
)
from aviary.backend.llm.utils import (
    init_torch_dist_process_group_async,
    initialize_node,
    timeit,
)
from aviary.backend.logger import get_logger
from aviary.backend.server.models import (
    LLMConfig,
    Prompt,
    Response,
    StaticBatchingModel,
)

WARMUP_PROMPT = "Write a short story."

initialize_node_remote = ray.remote(initialize_node)

logger = get_logger(__name__)


@timeit
def init_model(
    llm_config: LLMConfig,
    world_size: int,
    local_rank: int,
):
    """Initialize the model.

    Args:
        llm_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
        local_rank (int): Local rank of the current GPU.
    """
    logger.info(f"Initializing model {llm_config.model_id}...")

    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    initializer_name = llm_config.initialization.initializer
    if not isinstance(initializer_name, str):
        initializer_name = initializer_name.type
    initializer = get_initializer_cls_by_name(initializer_name)(
        device=device,
        world_size=world_size,
        **llm_config.initialization.initializer.get_initializer_kwargs(),
    )

    pipeline_name = llm_config.initialization.pipeline
    pipeline = get_pipeline_cls_by_name(pipeline_name).from_initializer(
        initializer,
        llm_config.actual_hf_model_id,
    )

    return pipeline


class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup will form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.

    Args:
        llm_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
    """

    def __init__(self, llm_config: LLMConfig, world_size: int):
        self.llm_config = llm_config
        self.world_size = world_size
        self.local_rank = None
        self.current_device = None
        self.gpu_memory_fraction = 1.0

    def init_model(
        self,
        local_rank: int,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: float = 0,
    ):
        """Initialize model for inference.

        Args:
            local_rank (int): Local rank of the current GPU.
            num_cpus_per_worker (int, optional): Number of CPUs to use per worker. Defaults to 1.
        """

        # Recreate the logger to make sure it takes precedence over
        # other logger configurations.
        get_logger(__name__, force=True)

        os.environ["OMP_NUM_THREADS"] = str(int(num_cpus_per_worker))
        expected_device = None
        if (
            torch.cuda.is_available()
            and num_gpus_per_worker > 0
            and num_gpus_per_worker < 1
        ):
            self.gpu_memory_fraction = num_gpus_per_worker
            expected_device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_per_process_memory_fraction(
                num_gpus_per_worker, device=expected_device
            )
        self.local_rank = local_rank

        self.generator = init_model(
            self.llm_config,
            self.world_size,
            self.local_rank,
        )
        if torch.cuda.is_available():
            # Save the current device so we can set it again if it gets reset
            self.current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            assert expected_device is None or self.current_device == expected_device

        self.warmup()

    def warmup(self):
        # Warmup
        # For DS w/ kernel inject, first batch the model gets MUST be of maximum batch size,
        # otherwise subsequent batches with more entries than the first batch
        # will raise CUDA errors if use_kernel=True.
        batch_size = self.llm_config.generation.max_batch_size or 1
        full_warmup = self.llm_config.initialization.full_warmup
        n_repeats = self.llm_config.generation.max_input_words if full_warmup else 1
        prompt = [WARMUP_PROMPT] * max(
            1, (int(n_repeats / (len(WARMUP_PROMPT.split()) + 1)))
        )
        prompt = " ".join(prompt)
        logger.info(
            f"Model {self.llm_config.model_id} is warming up, input len {len(prompt)}..."
        )
        generate_kwargs = self.llm_config.generation.all_generate_kwargs.copy()
        if "max_new_tokens" in generate_kwargs:
            if full_warmup:
                generate_kwargs["min_new_tokens"] = generate_kwargs["max_new_tokens"]
            else:
                generate_kwargs["max_new_tokens"] = generate_kwargs.get(
                    "min_new_tokens", 16
                )
        warmup_success = False
        while not warmup_success:
            try:
                assert batch_size > 0
                resp1 = self.generator(
                    [prompt] * batch_size,
                    **generate_kwargs,
                )
                logger.info(str(resp1))
                assert len(resp1) == batch_size
                assert all(x.generated_text for x in resp1)
                resp2 = self.generator(
                    [prompt],
                    **generate_kwargs,
                )
                logger.info(str(resp2))
                assert len(resp2) == 1
                assert all(x.generated_text for x in resp2)
                warmup_success = True
            except torch.cuda.OutOfMemoryError:
                batch_size -= 2
                logger.warning(
                    f"Warmup failed due to CUDA OOM, reducing batch size to {batch_size}"
                )

        logger.info(f"Model succesfully initialized, final batch size {batch_size}!")

        gc.collect()

    @timeit
    def stream(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        **kwargs,
    ) -> Iterator[List[Response]]:
        """Stream predictions using a Pipeline.

        Args:
            prompts (List[Prompt]): List of prompts.
            timeout_s (Optional[float], optional): Timeout for the generation.
                Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation. Ignored if timeout_s is None.
            **generate_kwargs: Keyword arguments to pass to the pipeline's `generate` method.
        """
        if self.current_device:
            torch.cuda.set_device(self.current_device)
        if not isinstance(self.generator, StreamingPipeline):
            raise RuntimeError(f"Pipeline {self.generator} does not support streaming.")
        prompt_text = [
            self.llm_config.generation.prompt_format.generate_prompt(p) for p in prompts
        ]
        yield from self.generator.stream(
            prompt_text,
            timeout_s=timeout_s,
            start_timestamp=start_timestamp,
            **kwargs,
        )

    @timeit
    async def async_stream(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        **kwargs,
    ) -> Iterator[List[Response]]:
        """Stream predictions asynchronously using a Pipeline.

        Args:
            prompts (List[Prompt]): List of prompts.
            timeout_s (Optional[float], optional): Timeout for the generation.
                Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation. Ignored if timeout_s is None.
            **kwargs: Keyword arguments to pass to the pipeline's `generate` method.
        """
        if self.current_device:
            torch.cuda.set_device(self.current_device)

        if not isinstance(self.generator, AsyncStreamingPipeline):
            raise RuntimeError(
                f"Pipeline {self.generator} does not support async streaming."
            )
        prompt_text = [
            self.llm_config.generation.prompt_format.generate_prompt(p) for p in prompts
        ]
        async for result in self.generator.async_stream(
            prompt_text,
            timeout_s=timeout_s,
            start_timestamp=start_timestamp,
            **kwargs,
        ):
            yield result

    @timeit
    def generate(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        oom_retry: bool = True,
        **kwargs,
    ) -> List[Response]:
        """Generate text from prompts.

        Args:
            prompts (List[Prompt]): Batch of prompts.
            timeout_s (Optional[float], optional): Timeout for the generation.
                Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation. Ignored if timeout_s is None.
            oom_retry (bool, optional): Whether to retry if CUDA OOM occurs.
            **kwargs: Keyword arguments to pass to the pipeline's `generate` method.
        """
        if self.current_device:
            torch.cuda.set_device(self.current_device)

        prompt_text = [
            self.llm_config.generation.prompt_format.generate_prompt(p) for p in prompts
        ]
        try:
            outputs = self.generator(
                prompt_text,
                timeout_s=timeout_s,
                start_timestamp=start_timestamp,
                **kwargs,
            )
            return outputs
        except torch.cuda.OutOfMemoryError as e:
            if not oom_retry:
                raise e
            else:
                logger.error(
                    "[FIXME] Prediction failed due to CUDA OOM, trying again...\n"
                    f"{traceback.format_exc()}"
                )
                prompt_text_1, prompt_text_2 = (
                    prompt_text[: len(prompt_text) // 2],
                    prompt_text[len(prompt_text) // 2 :],
                )
                responses_1 = self.generator(
                    prompt_text_1,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **kwargs,
                )
                responses_2 = self.generator(
                    prompt_text_2,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **kwargs,
                )
                return responses_1 + responses_2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}:{self.llm_config.model_id}"

    def ping(self) -> bool:
        """Ping the worker."""
        return True

    def can_stream(self) -> bool:
        """Whether the worker can stream."""
        return isinstance(self.generator, StreamingPipeline)

    def can_async_stream(self) -> bool:
        """Whether the worker can stream."""
        return isinstance(self.generator, AsyncStreamingPipeline)


class LLMPredictor:
    """Predictor for LLM models."""

    def __init__(self, model_config: Optional[StaticBatchingModel]) -> None:
        self.model_config = model_config
        self.base_worker_group = None
        self.new_worker_group = None
        self.can_stream = None
        self._base_worker_group_lock = asyncio.Lock()
        self._new_worker_group_lock = asyncio.Lock()

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
        return await init_torch_dist_process_group_async(worker_group, **kwargs)

    async def _start_prediction_workers(
        self, scaling_config: ScalingConfig, remote_prediction_worker_cls: type
    ):
        # Create the prediction workers.
        logger.info("Creating prediction workers...")
        worker_group = [
            remote_prediction_worker_cls.remote(
                self.model_config, scaling_config.num_workers
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
        return worker_group

    def _prepare_worker_runtime_env(self) -> dict:
        runtime_env = self.model_config.initialization.runtime_env or {}
        runtime_env.setdefault("env_vars", {})
        runtime_env["env_vars"].setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync"
        )
        return runtime_env

    async def _create_worker_group(
        self,
        scaling_config: ScalingConfig,
        pg_timeout_s: float = 600,
        prediction_worker_cls: Type[PredictionWorker] = PredictionWorker,
    ) -> List[ray.ObjectRef]:
        """Create a new worker group.

        Args:
            scaling_config (ScalingConfig): Scaling configuration for the new worker group.
            pg_timeout_s (float, optional): Timeout for the new worker group to be ready. Defaults to 600.
        """
        assert self.model_config
        llm_config = self.model_config

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
        remote_prediction_worker_cls = ray.remote(prediction_worker_cls).options(
            **scaling_options, runtime_env=runtime_env
        )
        initialize_node_remote_pg = initialize_node_remote.options(
            **scaling_options, runtime_env=runtime_env
        )

        logger.info("Waiting for placement group to be ready...")
        # This will raise a timeout error.
        await asyncio.wait_for(self.pg.ready(), timeout=pg_timeout_s)

        logger.info("Starting initialize_node tasks...")
        await asyncio.gather(
            *[
                initialize_node_remote_pg.remote(
                    llm_config.actual_hf_model_id,
                    llm_config.initialization.s3_mirror_config,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # Download just the tokenizer for the predictor
        initialize_node(
            llm_config.actual_hf_model_id,
            llm_config.initialization.s3_mirror_config,
            tokenizer_only=True,
        )

        worker_group = await self._start_prediction_workers(
            scaling_config=scaling_config,
            remote_prediction_worker_cls=remote_prediction_worker_cls,
        )

        self.can_stream = all(
            await asyncio.gather(*[worker_group[0].can_stream.remote()])
        )

        return worker_group

    async def _stream_async(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
    ) -> Iterator[List[Response]]:
        """Generate text for a list of prompts.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to 60. Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.

        Returns:
            A list of generated texts.
        """
        if self.can_stream:
            # async with self._base_worker_group_lock:
            tasks = [
                worker.stream.options(num_returns="streaming").remote(
                    prompts,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **self.model_config.generation.all_generate_kwargs,
                )
                for worker in self.base_worker_group
            ]
            async for result in tasks[0]:
                yield await result
        else:
            logger.warning(
                f"Pipeline {self.model_config.initialization.pipeline} does not support streaming. Ignoring queue."
            )
            yield await self._predict_async(
                prompts, timeout_s=timeout_s, start_timestamp=start_timestamp
            )

    async def _predict_async(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
    ) -> List[Response]:
        """Generate text for a list of prompts.

        Args:
            prompts (List[Prompt]): Batch of prompts to generate text from.
            timeout_s (float, optional): Timeout for the generation. Defaults
                to 60. Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation.

        Returns:
            A list of generated texts.
        """
        # async with self._base_worker_group_lock:
        prediction = (
            await asyncio.gather(
                *[
                    worker.generate.remote(
                        prompts,
                        timeout_s=timeout_s,
                        start_timestamp=start_timestamp,
                        **self.model_config.generation.all_generate_kwargs,
                    )
                    for worker in self.base_worker_group
                ]
            )
        )[0]
        return prediction

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

    def validate_prompt(self, prompt: Prompt) -> None:
        text = self.model_config.generation.prompt_format.generate_prompt(prompt)
        if len(text.split()) > self.model_config.generation.max_input_words:
            raise PromptTooLongError(
                f"Prompt exceeds max input words of "
                f"{self.model_config.generation.max_input_words}. "
                "Please make the prompt shorter."
            )
