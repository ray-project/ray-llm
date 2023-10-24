import asyncio
import logging

import ray
from transformers import AutoTokenizer

from rayllm.backend.llm.utils import initialize_node
from rayllm.backend.llm.vllm.vllm_models import VLLMApp, VLLMEngineConfig
from rayllm.backend.server.utils import make_async

logger = logging.getLogger(__name__)

initialize_node_remote = ray.remote(initialize_node)


class VLLMNodeInitializer:
    """Implements node initialization for VLLM.

    Runs the init node script on all relevant worker nodes.
    Also downloads the tokenizer to the local node.
    """

    async def initialize_node(self, llm_app: VLLMApp):
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

        logger.info("Starting initialize_node tasks on the workers and local node...")
        await asyncio.gather(
            *[
                initialize_node_remote_pg.remote(
                    engine_config.actual_hf_model_id,
                    engine_config.s3_mirror_config,
                    engine_config.gcs_mirror_config,
                )
                for i in range(scaling_config.num_workers)
            ]
        )

        # We can't do this in parallel because it could introduce a race where only the tokenizer is downloaded.
        # Each initialize node operation will skip if the node lock is acquired
        await self._initialize_local_node(engine_config)

        logger.info("Finished initialize_node tasks.")

        return pg, runtime_env

    @make_async
    def _initialize_local_node(self, engine_config: VLLMEngineConfig):
        local_path = initialize_node(
            engine_config.actual_hf_model_id,
            engine_config.s3_mirror_config,
            engine_config.gcs_mirror_config,
            tokenizer_only=True,
        )

        # Validate that the binary exists
        if local_path and local_path != engine_config.actual_hf_model_id:
            engine_config.hf_model_id = local_path

        # Download the tokenizer if it isn't a local file path
        _ = AutoTokenizer.from_pretrained(engine_config.actual_hf_model_id)
