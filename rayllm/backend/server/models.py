import copy
import json
import logging
import os
import time
from enum import Enum, IntEnum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml
from markdown_it import MarkdownIt
from pydantic import (
    BaseModel,
    Extra,
    Field,
    PrivateAttr,
    conlist,
    root_validator,
    validator,
)
from ray.air import ScalingConfig as AIRScalingConfig
from ray.serve.config import AutoscalingConfig
from ray.util.placement_group import (
    PlacementGroup,
    get_current_placement_group,
    placement_group_table,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rayllm.backend.llm.dict_utils import merge_dicts
from rayllm.backend.llm.error_handling import TooManyStoppingSequences
from rayllm.common.models import (
    DisabledPromptFormat,
    ErrorResponse,
    LogProbs,
    Message,
    Prompt,
    PromptFormat,
    Tool,
    ToolCall,
    ToolChoice,
)

# noqa
from rayllm.conf import ENV_VARS_TO_PROPAGATE
from rayllm.env_conf import (
    ALLOW_NEW_PLACEMENT_GROUPS_IN_DEPLOYMENT,
    MAX_NUM_STOPPING_SEQUENCES,
)

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)
logger = logging.getLogger(__name__)


class QueuePriority(IntEnum):
    """Lower value = higher priority"""

    GENERATE_TEXT = 0
    BATCH_GENERATE_TEXT = 1


def markdown_extract_first_paragraph(markdown_text: str) -> str:
    """Extract the first paragraph from a markdown-formatted string."""
    from mdit_py_plugins.front_matter import front_matter_plugin

    md = MarkdownIt("commonmark", {"breaks": True, "html": True}).use(
        front_matter_plugin
    )
    tokens = md.parse(markdown_text)
    first_paragraph: List[str] = []
    in_paragraph = False
    for token in tokens:
        if in_paragraph and token.tag == "p":
            in_paragraph = False
            if first_paragraph:
                break
            continue
        if in_paragraph:
            # Ignore images
            if token.children and token.children[0].type == "image":
                continue
            if token.content:
                first_paragraph.append(token.content)
        elif token.tag == "p":
            in_paragraph = True
    return "".join(first_paragraph).strip()


class BaseModelExtended(BaseModel):
    @classmethod
    def parse_yaml(cls: Type[ModelT], file, **kwargs) -> ModelT:
        kwargs.setdefault("Loader", yaml.SafeLoader)
        dict_args = yaml.load(file, **kwargs)
        return cls.parse_obj(dict_args)

    def yaml(
        self,
        *,
        stream=None,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: Union[bool, None] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **kwargs,
    ):
        """
        Generate a YAML representation of the model, `include` and `exclude` arguments as per `dict()`.
        """
        return yaml.dump(
            json.loads(
                self.json(
                    include=include,
                    exclude=exclude,
                    by_alias=by_alias,
                    skip_defaults=skip_defaults,
                    exclude_unset=exclude_unset,
                    exclude_defaults=exclude_defaults,
                    exclude_none=exclude_none,
                )
            ),
            stream=stream,
            **kwargs,
        )


class ComputedPropertyMixin:
    """
    Include properties in the dict and json representations of the model.
    """

    # Replace with pydantic.computed_field once it's available
    @classmethod
    def get_properties(cls):
        return [prop for prop in dir(cls) if isinstance(getattr(cls, prop), property)]

    def dict(self, *args, **kwargs):
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )
        return super().dict(*args, **kwargs)  # type: ignore

    def json(
        self,
        *args,
        **kwargs,
    ) -> str:
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )

        return super().json(*args, **kwargs)  # type: ignore


class AviaryModelResponse(ComputedPropertyMixin, BaseModelExtended):
    """The response from a query to a RayLLM Model.

    Args:
        generated_text: The generated text.
        tool_calls: The tool calls that were made.
        embedding_outputs: The embedding outputs.
        logprobs: Log probabilities of each token and possibly some of the unchosen tokens.
        num_input_tokens: The number of input tokens.
        num_generated_tokens: The number of generated tokens.
        num_input_tokens_batch: The number of input tokens in the batch.
        num_generated_tokens_batch: The number of generated tokens in the batch.
        preprocessing_time: The time spent preprocessing the request.
        generation_time: The time spent generating the response.
        timestamp: The timestamp of the response.
        finish_reason: The reason the generation finished.
        error: The error, if any.

    """

    generated_text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    embedding_outputs: Optional[Union[List[float], List[List[float]]]] = None
    logprobs: Optional[List[LogProbs]] = None
    num_input_tokens: Optional[int] = None
    num_input_tokens_batch: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    num_generated_tokens_batch: Optional[int] = None
    preprocessing_time: Optional[float] = None
    generation_time: Optional[float] = None
    timestamp: Optional[float] = Field(default_factory=time.time)
    finish_reason: Optional[str] = None
    error: Optional[ErrorResponse] = None

    @root_validator
    def text_or_error_or_finish_reason(cls, values):
        if (
            values.get("generated_text") is None
            and values.get("embedding_outputs") is None
            and values.get("error") is None
            and values.get("finish_reason") is None
        ):
            raise ValueError(
                "Either 'generated_text' or 'embedding_outputs' or 'error' or 'finish_reason' must be set"
            )
        return values

    @classmethod
    def merge_stream(cls, *responses: "AviaryModelResponse") -> "AviaryModelResponse":
        """
        Merge a stream of responses into a single response.

        The generated text is concatenated. Fields are maxed, except for
        num_generated_tokens and generation_time, which are summed.
        """
        if len(responses) == 1:
            return responses[0]

        generated_text = (
            None
            if responses[0].generated_text is None
            else "".join([response.generated_text or "" for response in responses])
        )
        embedding_outputs = (
            None
            if responses[0].embedding_outputs is None
            else [
                item
                for sublist in [
                    response.embedding_outputs or [] for response in responses
                ]
                for item in sublist
            ]
        )
        num_input_tokens = [
            response.num_input_tokens
            for response in responses
            if response.num_input_tokens is not None
        ]
        max_num_input_tokens = max(num_input_tokens) if num_input_tokens else None
        num_input_tokens_batch = [
            response.num_input_tokens_batch
            for response in responses
            if response.num_input_tokens_batch is not None
        ]
        max_num_input_tokens_batch = (
            max(num_input_tokens_batch) if num_input_tokens_batch else None
        )
        num_generated_tokens = [
            response.num_generated_tokens
            for response in responses
            if response.num_generated_tokens is not None
        ]
        total_generated_tokens = (
            sum(num_generated_tokens) if num_generated_tokens else None
        )
        num_generated_tokens_batch = [
            response.num_generated_tokens_batch
            for response in responses
            if response.num_generated_tokens_batch is not None
        ]
        total_generated_tokens_batch = (
            sum(num_generated_tokens_batch) if num_generated_tokens_batch else None
        )
        preprocessing_time = [
            response.preprocessing_time
            for response in responses
            if response.preprocessing_time is not None
        ]
        max_preprocessing_time = max(preprocessing_time) if preprocessing_time else None
        generation_time = [
            response.generation_time
            for response in responses
            if response.generation_time is not None
        ]
        total_generation_time = sum(generation_time) if generation_time else None
        error = next(
            (response.error for response in reversed(responses) if response.error), None
        )
        logprobs = []
        for response in responses:
            if response.logprobs:
                logprobs.extend(response.logprobs)

        return cls(
            generated_text=generated_text,
            embedding_outputs=embedding_outputs,
            logprobs=logprobs,
            num_input_tokens=max_num_input_tokens,
            num_input_tokens_batch=max_num_input_tokens_batch,
            num_generated_tokens=total_generated_tokens,
            num_generated_tokens_batch=total_generated_tokens_batch,
            preprocessing_time=max_preprocessing_time,
            generation_time=total_generation_time,
            timestamp=responses[-1].timestamp,
            finish_reason=responses[-1].finish_reason,
            error=error,
        )

    @property
    def total_time(self) -> Optional[float]:
        if self.generation_time is None and self.preprocessing_time is None:
            return None
        return (self.preprocessing_time or 0) + (self.generation_time or 0)

    @property
    def num_total_tokens(self) -> Optional[float]:
        try:
            return (self.num_input_tokens or 0) + (self.num_generated_tokens or 0)
        except Exception:
            return None

    @property
    def num_total_tokens_batch(self) -> Optional[float]:
        try:
            return (self.num_input_tokens_batch or 0) + (
                self.num_generated_tokens_batch or 0
            )
        except Exception:
            return None

    def unpack(self) -> Tuple["AviaryModelResponse", ...]:
        return (self,)


class BatchedAviaryModelResponse(AviaryModelResponse):
    # Same as AviaryModelResponse, but persists the individual responses
    # that were batched together to produce this response.

    _individual_responses: Optional[List[AviaryModelResponse]] = PrivateAttr(None)

    @classmethod
    def merge_stream(cls, *responses: "AviaryModelResponse") -> "AviaryModelResponse":
        if len(responses) == 1:
            return responses[0]
        obj = super().merge_stream(*responses)
        obj._individual_responses = list(responses)  # type: ignore
        return obj

    def unpack(self) -> Tuple["AviaryModelResponse", ...]:
        return tuple(self._individual_responses or [])


class S3AWSCredentials(BaseModelExtended):
    create_aws_credentials_url: str
    auth_token_env_variable: Optional[str]


class ExtraFiles(BaseModelExtended):
    bucket_uri: str
    destination_path: str


class S3MirrorConfig(BaseModelExtended):
    bucket_uri: Optional[str] = None
    s3_sync_args: Optional[List[str]] = None
    s3_aws_credentials: Optional[S3AWSCredentials] = None
    extra_files: List[ExtraFiles] = Field(default_factory=list)


class GCSMirrorConfig(BaseModelExtended):
    bucket_uri: str
    extra_files: List[ExtraFiles] = Field(default_factory=list)

    @validator("bucket_uri")
    def check_uri_format(cls, value: str):
        if not value.startswith("gs://"):
            raise ValueError(
                f'Got invalid value "{value}" for bucket_uri. '
                'Expected a URI that starts with "gs://".'
            )
        return value


class GenerationConfig(BaseModelExtended):
    prompt_format: Optional[Union[PromptFormat, DisabledPromptFormat]] = None
    generate_kwargs: Dict[str, Any] = {}
    stopping_sequences: Optional[List[Union[str, int, List[Union[str, int]]]]] = None

    @validator("prompt_format")
    def default_prompt_format(cls, prompt_format):
        return prompt_format if prompt_format is not None else DisabledPromptFormat()

    @validator("stopping_sequences")
    def check_stopping_sequences(cls, value):
        def try_int(x):
            if isinstance(x, list):
                return [try_int(y) for y in x]
            try:
                return int(x)
            except Exception:
                return x

        if value:
            value = try_int(value)
        return value

    @property
    def all_generate_kwargs(self) -> Dict[str, Any]:
        return {"stopping_sequences": self.stopping_sequences, **self.generate_kwargs}


class EngineType(str, Enum):
    """Engine type for the serve logs."""

    VLLMEngine = "VLLMEngine"
    EmbeddingEngine = "EmbeddingEngine"
    TRTLLMEngine = "TRTLLMEngine"


class ModelType(str, Enum):
    text_generation = "text-generation"
    text_generation_finetuned = "text-generation-finetuned"
    embedding = "embedding"


class EngineConfig(BaseModelExtended):
    class Config:
        use_enum_values = True
        extra = Extra.forbid

    model_id: str
    hf_model_id: Optional[str] = None
    type: EngineType
    model_type: ModelType
    tokenizer_id: Optional[str]

    s3_mirror_config: Optional[S3MirrorConfig] = None
    gcs_mirror_config: Optional[GCSMirrorConfig] = None
    engine_kwargs: Dict[str, Any] = {}

    max_total_tokens: int = 2048

    runtime_env: Optional[Dict[str, Any]] = None

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

    def get_vllm_load_s3_path(self) -> Optional[str]:
        if self.type == EngineType.VLLMEngine:
            return self.engine_kwargs.get("load_s3_path", None)
        return None


class SchedulingMetadata(BaseModelExtended):
    request_id: Union[str, List[str]]
    priority: Union[QueuePriority, List[QueuePriority]]

    class Config:
        arbitrary_types_allowed = True


class SamplingParams(BaseModelExtended):
    """
    Args:
        max_tokens: The maximum number of tokens to generate. Defaults to inf.
        temperature: What sampling temperature to use.
        top_p: An alternative to sampling with temperature, called nucleus sampling.
        n: How many completions to generate for each prompt.
        logprobs: Include the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens.
        top_logprobs: The number of logprobs to return. Defaults to 1. `logprobs`
            must be set to `True` in order to use top_logprobs.
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
        response_format: Format to return the final response in. Can be for ex:
            response_format={"type": "json", "schema": "{...}"}

    """

    _ignored_fields: Set[str] = set()

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: int = 1
    response_format: Optional[Dict[str, Any]] = None

    def dict(self, **kwargs):
        if kwargs.get("exclude", None) is None:
            kwargs["exclude"] = self._ignored_fields
        return super().dict(**kwargs)

    @validator("stop", always=True)
    def validate_stopping_sequences(cls, values):
        if not values:
            return values

        unique_val = sorted(list(set(values)))

        if len(unique_val) > MAX_NUM_STOPPING_SEQUENCES:
            TooManyStoppingSequences(
                len(unique_val), MAX_NUM_STOPPING_SEQUENCES
            ).raise_exception()

        return unique_val

    @classmethod
    def merge_generation_params(
        cls: Type[ModelT], prompt: Prompt, generation: GenerationConfig
    ) -> ModelT:
        # Extract parameters object from prompt
        parameters = prompt.parameters or {}
        if not isinstance(parameters, dict):
            parameters = parameters.dict()

        # Merge in the generate kwargs
        generate_kwargs_copy = copy.deepcopy(generation.generate_kwargs)
        generate_kwargs = merge_dicts(
            generate_kwargs_copy,
            parameters,
        )

        # The stoppping sequence needs to be merged manually
        generate_kwargs["stop"] = (parameters.get("stop") or []) + (
            generation.stopping_sequences or []
        )

        return cls.parse_obj(generate_kwargs)


class GenerationRequest(BaseModelExtended):
    prompt: Union[str, List[int], List[str]]
    request_id: Union[str, List[str]]
    sampling_params: Optional[Union[SamplingParams, List[SamplingParams]]]
    scheduling_metadata: Union[SchedulingMetadata, List[SchedulingMetadata]]


class ChatCompletions(BaseModelExtended):
    """
    Args:
        model: The model to query.
        messages: A list of messages describing the conversation so far.
            Contains a required "role", which is the role of the author of this
            message. One of "system", "user", or "assistant".
            Also contains required "content", the contents of the message, and
            an optional "name", the name of the author of this message.
        stream: Whether to stream back partial progress.
        echo: Echo back the prompt in addition to the completion.
        user: A unique identifier representing your end-user, which can help us
            to monitor and detect abuse. Learn more.
    """

    model: str
    messages: conlist(Message, min_items=1)
    stream: bool = False
    echo: Optional[bool] = False
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Union[Literal["auto", "none"], ToolChoice] = "auto"


class Completions(BaseModelExtended):
    """
    Args:
        model: The model to query.
        prompt: The prompt to generate completions for, encoded as string.
        suffix: The suffix that comes after a completion of inserted text.
        stream: Whether to stream back partial progress.
        echo: Echo back the prompt in addition to the completion.
        user: A unique identifier representing your end-user, which can help us
            to monitor and detect abuse. Learn more.
    """

    model: str
    prompt: str
    suffix: Optional[str] = None
    stream: bool = False
    echo: Optional[bool] = False
    user: Optional[str] = None


class Embeddings(BaseModelExtended):
    """
    Args:
        model: the mode to query.
        input: the input to generate embeddings for, encoded as a string.
        user: A unique identifier representing the end-user, which helps monitoring and abuse detection.
    """

    model: str
    input: Union[str, conlist(str, min_items=1)]
    user: Optional[str] = None


class ScalingConfig(BaseModelExtended):
    num_workers: int
    num_gpus_per_worker: float = 1
    num_cpus_per_worker: float = 1
    placement_strategy: str = "PACK"
    resources_per_worker: Optional[Dict[str, float]] = None
    pg_timeout_s: float = 600

    @validator("num_gpus_per_worker")
    def validate_num_gpus_per_worker(cls, value):
        if value > 1:
            raise ValueError(
                f"num_gpus_per_worker must be <= 1, got {value}. "
                "If you want to use multiple GPUs, change num_workers instead."
            )
        return value

    def as_air_scaling_config(self) -> "AIRScalingConfig":
        return AIRScalingConfig(
            use_gpu=self.num_gpus_per_worker > 0,
            num_workers=self.num_workers,
            trainer_resources={"CPU": 0},
            resources_per_worker={
                "CPU": self.num_cpus_per_worker,
                "GPU": self.num_gpus_per_worker,
                **(self.resources_per_worker or {}),
            },
            placement_strategy=self.placement_strategy,
        )


class Args(BaseModelExtended):
    engine_config: EngineConfig
    scaling_config: ScalingConfig

    @property
    def air_scaling_config(self) -> AIRScalingConfig:
        return self.scaling_config.as_air_scaling_config()


class ServeMultiplexConfig(BaseModelExtended):
    max_num_models_per_replica: int


class DeploymentConfig(BaseModelExtended):
    autoscaling_config: Optional[AutoscalingConfig]
    max_concurrent_queries: Optional[int] = None
    ray_actor_options: Optional[Dict[str, Any]] = None


class LLMApp(Args):
    """The full configuration of a single LLM Model"""

    deployment_config: Optional[DeploymentConfig] = None
    multiplex_config: Optional[ServeMultiplexConfig] = None
    enabled: bool = True

    @property
    def model_id(self):
        return self.engine_config.model_id

    def short_metadata(self):
        return self.dict(
            include={
                "model_id": True,
                "engine_config": {
                    "model_type",
                    "generation",
                    "model_url",
                    "max_total_tokens",
                    "model_description",
                },
            }
        )

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
        else:
            if not ALLOW_NEW_PLACEMENT_GROUPS_IN_DEPLOYMENT:
                raise RuntimeError(
                    "Creating new placement groups is not allowed. "
                    "Change RAYLLM_ALLOW_NEW_PLACEMENT_GROUPS_IN_DEPLOYMENT "
                    "if this is not intended."
                )
            pg = (
                self.air_scaling_config.as_placement_group_factory().to_placement_group()
            )
            logger.info(f"Using new placement group {pg}. {placement_group_table(pg)}")
        return pg


class ServeArgs(BaseModel):
    models: Union[str, LLMApp, List[Union[str, LLMApp]]]


class AppArgs(BaseModel):
    model: Union[str, LLMApp]


class RouterArgs(BaseModel):
    models: Union[str, LLMApp, List[Union[LLMApp, str]]] = None
    embedding_models: Union[str, LLMApp, List[Union[str, LLMApp]]] = None
    trtllm_models: Union[str, LLMApp, List[Union[LLMApp, str]]] = None


class PlacementConfig(BaseModel):
    world_size: int
    scaling_config: ScalingConfig


class HasModelId(Protocol):
    model_id: str


class ChatCompletionsParams(ChatCompletions, SamplingParams, extra=Extra.allow):
    pass


class CompletionsParams(Completions, SamplingParams, extra=Extra.allow):
    max_tokens: Optional[int] = 16
