import asyncio
import logging
from typing import Union

import ray
from transformers import AutoTokenizer

from rayllm.backend.llm.embedding.embedding_models import EmbeddingEngineConfig
from rayllm.backend.llm.utils import initialize_node
from rayllm.backend.llm.vllm.vllm_models import VLLMEngineConfig
from rayllm.backend.server.models import LLMApp, ScalingConfig
from rayllm.backend.server.utils import make_async

logger = logging.getLogger(__name__)

initialize_node_remote = ray.remote(initialize_node)


class LLMNodeInitializer:
    """Implements node initialization for LLM.

    Runs the init node script on all relevant worker nodes.
    Also downloads the tokenizer to the local node.
    """

    def __init__(self, local_node_tokenizer_only: bool):
        self.local_node_tokenizer_only = local_node_tokenizer_only

    async def initialize_node(self, llm_app: LLMApp):
        engine_config = llm_app.engine_config
        scaling_config = llm_app.scaling_config

        pg = llm_app.get_or_create_pg()
        scaling_options = llm_app.get_scaling_options(pg)

        # Get the runtime env
        runtime_env = llm_app.engine_config.get_runtime_env_with_local_env_vars()

        # Initialize the node
        initialize_node_remote_pg = initialize_node_remote.options(
            **scaling_options, runtime_env=runtime_env
        )

        # This is needed for checking the cache path when both mirror config and cache is defined
        num_workers = scaling_config.num_workers
        model_s3_cache = engine_config.get_vllm_load_s3_path()

        logger.info("Starting initialize_node tasks on the workers and local node...")
        await asyncio.gather(
            *[
                initialize_node_remote_pg.remote(
                    engine_config.actual_hf_model_id,
                    engine_config.s3_mirror_config,
                    engine_config.gcs_mirror_config,
                    model_s3_cache=model_s3_cache,
                    num_workers=num_workers,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # We can't do this in parallel because it could introduce a race where only the tokenizer is downloaded.
        # Each initialize node operation will skip if the node lock is acquired
        await self._initialize_local_node(engine_config, scaling_config)

        logger.info("Finished initialize_node tasks.")

        return pg, runtime_env

    @make_async
    def _initialize_local_node(
        self,
        engine_config: Union[VLLMEngineConfig, EmbeddingEngineConfig],
        scaling_config: ScalingConfig,
    ):
        # This is needed for checking the cache path when both mirror config and cache is defined
        num_workers = scaling_config.num_workers
        model_s3_cache = engine_config.get_vllm_load_s3_path()

        local_path = initialize_node(
            engine_config.actual_hf_model_id,
            engine_config.s3_mirror_config,
            engine_config.gcs_mirror_config,
            tokenizer_only=self.local_node_tokenizer_only,
            model_s3_cache=model_s3_cache,
            num_workers=num_workers,
        )

        # Validate that the binary exists
        if local_path and local_path != engine_config.actual_hf_model_id:
            engine_config.hf_model_id = local_path

        logger.info(f"Downloading the tokenizer for {engine_config.actual_hf_model_id}")
        # Download the tokenizer if it isn't a local file path
        _ = AutoTokenizer.from_pretrained(engine_config.actual_hf_model_id)
