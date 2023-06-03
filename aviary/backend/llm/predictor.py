import asyncio
import gc
import os
import traceback
from typing import List, Optional

import ray
import ray.util
import torch
import torch.backends.cuda
from ray.air import ScalingConfig
from ray.air.util.torch_dist import TorchDistributedWorker
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from aviary.backend.llm.initializers import get_initializer_cls_by_name
from aviary.backend.llm.pipelines import get_pipeline_cls_by_name
from aviary.backend.llm.pipelines._base import BasePipeline
from aviary.backend.llm.utils import (
    init_torch_dist_process_group_async,
    initialize_node,
    timeit,
)
from aviary.backend.logger import get_logger
from aviary.backend.server.models import Args, LLMConfig, Prompt, Response

WARMUP_PROMPT = "Write a short story."

initialize_node_remote = ray.remote(initialize_node)

logger = get_logger(__name__)


@timeit
def init_model(
    llm_config: LLMConfig,
    world_size: int,
    local_rank: int,
    max_batch_size: Optional[int] = None,
):
    """Initialize the model.

    Args:
        llm_config (LLM): LLM configuration.
        world_size (int): Number of GPUs.
        local_rank (int): Local rank of the current GPU.
        max_batch_size (Optional[int], optional): Maximum batch size. Defaults to None.
    """
    logger.info(f"Initializing model {llm_config.model_id}...")

    # Lazy import so that the new cache location is used
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
        prompt_format=llm_config.generation.prompt_format,
    )

    # Warmup
    # For DS w/ kernel inject, first batch the model gets MUST be of maximum batch size,
    # otherwise subsequent batches with more entries than the first batch
    # will raise CUDA errors if use_kernel=True.
    batch_size = max_batch_size or 1
    prompt = [WARMUP_PROMPT] * (
        int(llm_config.max_input_words / (len(WARMUP_PROMPT.split()) + 1)) + 1
    )
    prompt = " ".join(prompt)
    logger.info(
        f"Model {llm_config.model_id} is warming up, input len {len(prompt)}..."
    )
    generate_kwargs = llm_config.generation.all_generate_kwargs.copy()
    if "max_new_tokens" in generate_kwargs:
        generate_kwargs["min_new_tokens"] = generate_kwargs["max_new_tokens"]
    warmup_success = False
    while not warmup_success:
        try:
            assert batch_size > 0
            resp1 = generate(
                [prompt] * batch_size,
                pipeline,
                **generate_kwargs,
            )
            logger.info(str(resp1))
            assert len(resp1) == batch_size
            assert all(x.generated_text for x in resp1)
            resp2 = generate(
                [prompt],
                pipeline,
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

    logger.info(
        f"Model {llm_config.model_id} succesfully initialized, final batch size {batch_size}!"
    )

    gc.collect()

    return pipeline


@timeit
def generate(
    prompts: List[Prompt], pipeline: BasePipeline, **generate_kwargs
) -> List[Response]:
    """Generate predictions using a Pipeline.

    Args:
        prompts (List[Prompt]): List of prompts.
        pipeline (BasePipeline): Pipeline to use.
        **generate_kwargs: Keyword arguments to pass to the pipeline's `generate` method.
    """
    outputs = pipeline(
        prompts,
        **generate_kwargs,
    )
    return outputs


@ray.remote
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

    def init_model(
        self,
        local_rank: int,
        num_cpus_per_worker: int = 1,
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

        self.generator = init_model(
            self.llm_config,
            self.world_size,
            local_rank,
            max_batch_size=self.llm_config.generation.max_batch_size,
        )

    def generate(
        self,
        data: List[Prompt],
        *,
        timeout_s: Optional[float] = None,
        start_timestamp: Optional[float] = None,
        oom_retry: bool = True,
        **kwargs,
    ) -> List[str]:
        """Generate text from prompts.

        Args:
            data (List[Prompt]): Batch of prompts.
            timeout_s (Optional[float], optional): Timeout for the generation.
                Ignored if start_timestamp is None.
            start_timestamp (Optional[float], optional): Timestamp of when the
                batch was created. Defaults to None. If set, will early stop
                the generation. Ignored if timeout_s is None.
            oom_retry (bool, optional): Whether to retry if CUDA OOM occurs.
        """
        try:
            return generate(
                data,
                self.generator,
                timeout_s=timeout_s,
                start_timestamp=start_timestamp,
                **kwargs,
            )
        except torch.cuda.OutOfMemoryError as e:
            if not oom_retry:
                raise e
            else:
                logger.error(
                    "[FIXME] Prediction failed due to CUDA OOM, trying again...\n"
                    f"{traceback.format_exc()}"
                )
                data_1, data_2 = data[: len(data) // 2], data[len(data) // 2 :]
                responses_1 = generate(
                    data_1,
                    self.generator,
                    timeout_s=timeout_s,
                    start_timestamp=start_timestamp,
                    **kwargs,
                )
                responses_2 = generate(
                    data_2,
                    self.generator,
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


class LLMPredictor:
    """Predictor for LLM models."""

    def __init__(self) -> None:
        self.base_worker_group = None
        self.new_worker_group = None
        self._base_worker_group_lock = asyncio.Lock()
        self._new_worker_group_lock = asyncio.Lock()

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
            async with self._base_worker_group_lock:
                logger.info(f"Rolling over to new worker group {self.new_worker_group}")
                self.base_worker_group = self.new_worker_group
                self.new_worker_group = None
            gc.collect()

    async def _create_worker_group(
        self, scaling_config: ScalingConfig, pg_timeout_s: float = 600
    ) -> List[ray.ObjectRef]:
        """Create a new worker group.

        Args:
            scaling_config (ScalingConfig): Scaling configuration for the new worker group.
            pg_timeout_s (float, optional): Timeout for the new worker group to be ready. Defaults to 600.
        """
        gc.collect()

        config: Args = self.args
        llm_config = config.model_config

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
        runtime_env = llm_config.initialization.runtime_env or {}
        prediction_worker_cls = PredictionWorker.options(
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
                    llm_config.model_id,
                    llm_config.initialization.s3_mirror_config,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # Create the prediction workers.
        logger.info("Creating prediction workers...")
        worker_group = [
            prediction_worker_cls.remote(llm_config, scaling_config.num_workers)
            for i in range(scaling_config.num_workers)
        ]

        logger.info("Initializing torch_dist process group on workers...")
        # Initialize torch distributed process group for the workers.
        local_ranks = await init_torch_dist_process_group_async(
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
                )
                for worker, local_rank in zip(worker_group, local_ranks)
            ]
        )

        return worker_group

    async def _predict_async(
        self,
        prompts: List[Prompt],
        *,
        timeout_s: float = 60,
        start_timestamp: Optional[float] = None,
    ) -> List[str]:
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
        async with self._base_worker_group_lock:
            prediction = (
                await asyncio.gather(
                    *[
                        worker.generate.remote(
                            prompts,
                            timeout_s=timeout_s,
                            start_timestamp=start_timestamp,
                            **self.args.model_config.generation.all_generate_kwargs,
                        )
                        for worker in self.base_worker_group
                    ]
                )
            )[0]
        return prediction
