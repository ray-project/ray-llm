import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import validator
from ray.util.placement_group import (
    PlacementGroup,
    get_current_placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rayllm.backend.server.models import (
    BaseModelExtended,
    ChatCompletions,
    Completions,
    EngineConfig,
    GCSMirrorConfig,
    GenerationConfig,
    LLMApp,
    PlacementConfig,
    S3MirrorConfig,
    SamplingParams,
    SchedulingMetadata,
)
from rayllm.conf import ENV_VARS_TO_PROPAGATE

logger = logging.getLogger(__name__)


class VLLMEngineConfig(EngineConfig):
    type: Literal["VLLMEngine"] = "VLLMEngine"
    model_id: str
    model_url: Optional[str] = None
    model_description: Optional[str] = None
    generation: GenerationConfig

    s3_mirror_config: Optional[S3MirrorConfig] = None
    gcs_mirror_config: Optional[GCSMirrorConfig] = None
    runtime_env: Optional[Dict[str, Any]] = None
    hf_model_id: Optional[str] = None
    engine_kwargs: Dict[str, Any] = {}

    max_total_tokens: int = 2048

    # The environment variables to propogate to the workers
    # These will be copied to the runtime env
    env_vars_to_propogate: List[str] = list(ENV_VARS_TO_PROPAGATE)

    @validator("gcs_mirror_config")
    def check_only_one_mirror_config_specified(cls, value, values):
        gcs_config = value
        s3_config = values["s3_mirror_config"]

        if gcs_config is not None and s3_config is not None:
            raise ValueError(
                "Both s3_mirror_config and gcs_mirror_config were specified. "
                "Only one of these can be specified. Please set one of them "
                "to None."
            )

        return value

    @property
    def actual_hf_model_id(self) -> str:
        return self.hf_model_id or self.model_id

    @property
    def sampling_params_model(self):
        return VLLMSamplingParams

    @property
    def completions_model(self):
        return VLLMCompletions

    @property
    def chat_completions_model(self):
        return VLLMChatCompletions

    def get_initialization_kwargs(self) -> dict:
        """
        Get kwargs that will be actually passed to the LLMInitializer
        constructor.
        """
        return self.engine_kwargs.copy()

    def get_runtime_env_with_local_env_vars(self) -> dict:
        runtime_env = self.runtime_env or {}
        runtime_env.setdefault("env_vars", {})

        # Propogate env vars to the runtime env
        for env_var in self.env_vars_to_propogate:
            if env_var in os.environ:
                runtime_env["env_vars"][env_var] = os.getenv(env_var)
        return runtime_env


class VLLMSamplingParams(SamplingParams):
    """
    Args:
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """

    _ignored_fields = {"best_of", "n", "logit_bias", "logprobs"}

    top_k: Optional[int] = None


class VLLMChatCompletions(ChatCompletions, VLLMSamplingParams):
    """
    Args:
        model: The model to query.
        messages: A list of messages describing the conversation so far.
            Contains a required "role", which is the role of the author of this
            message. One of "system", "user", or "assistant".
            Also contains required "content", the contents of the message, and
            an optional "name", the name of the author of this message.
        max_tokens: The maximum number of tokens to generate. Defaults to inf.
        temperature: What sampling temperature to use.
        top_p: An alternative to sampling with temperature, called nucleus sampling.
        n: How many completions to generate for each prompt.
        stream: Whether to stream back partial progress.
        logprobs: Include the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens.
        echo: Echo back the prompt in addition to the completion.
        stop: Up to 4 sequences where the API will stop generating further tokens.
            The returned text will not contain the stop sequence.
        presence_penalty: Number between -2.0 and 2.0.
            Positive values penalize new tokens based on whether they appear in
            the text so far, increasing the model's likelihood to talk about
            new topics.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on their existing frequency in the text so far,
            decreasing the model's likelihood to repeat the same line verbatim.
        best_of: Generates `best_of` completions server-side and returns the "best".
        logit_bias: Modify the likelihood of specified tokens appearing in
            the completion.
        user: A unique identifier representing your end-user, which can help us
            to monitor and detect abuse. Learn more.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """

    pass


class VLLMCompletions(Completions, VLLMSamplingParams):
    """
    Args:
        model: The model to query.
        prompt: The prompt to generate completions for, encoded as string.
        suffix: The suffix that comes after a completion of inserted text.
        max_tokens: The maximum number of tokens to generate.
        temperature: What sampling temperature to use.
        top_p: An alternative to sampling with temperature, called nucleus sampling.
        n: How many completions to generate for each prompt.
        stream: Whether to stream back partial progress.
        logprobs: Include the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens.
        echo: Echo back the prompt in addition to the completion.
        stop: Up to 4 sequences where the API will stop generating further tokens.
            The returned text will not contain the stop sequence.
        presence_penalty: Number between -2.0 and 2.0.
            Positive values penalize new tokens based on whether they appear in
            the text so far, increasing the model's likelihood to talk about
            new topics.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on their existing frequency in the text so far,
            decreasing the model's likelihood to repeat the same line verbatim.
        best_of: Generates `best_of` completions server-side and returns the "best".
        logit_bias: Modify the likelihood of specified tokens appearing in
            the completion.
        user: A unique identifier representing your end-user, which can help us
            to monitor and detect abuse. Learn more.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    """

    max_tokens: Optional[int] = 16


class VLLMGenerationRequest(BaseModelExtended):
    prompt: Union[str, List[int]]
    request_id: str
    sampling_params: VLLMSamplingParams
    scheduling_metadata: SchedulingMetadata


class VLLMApp(LLMApp):
    engine_config: VLLMEngineConfig  # type: ignore

    @property
    def placement_config(self):
        return PlacementConfig(
            world_size=self.scaling_config.num_workers,
            scaling_config=self.scaling_config,
        )

    def get_scaling_options(self, pg: Optional[PlacementGroup] = None):
        """Get AIR scaling configs"""
        scaling_config = self.air_scaling_config
        pg = pg or self.get_or_create_pg()
        return dict(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            resources=scaling_config.additional_resources_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_capture_child_tasks=True
            ),
        )

    def get_or_create_pg(self):
        """Get or a create a placement group

        If we are already in a placement group, return the existing placement group.
        Else, create a new placement group based on the scaling config
        """
        pg = get_current_placement_group()
        if pg:
            logger.info(
                f"Using existing placement group {pg} {pg.id}. {placement_group_table(pg)}"
            )
            logger.info(f"Using existing placement group {pg}")
        else:
            pg = (
                self.air_scaling_config.as_placement_group_factory().to_placement_group()
            )
            logger.info(f"Using new placement group {pg}. {placement_group_table(pg)}")
        return pg
