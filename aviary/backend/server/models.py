import time
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import yaml
from markdown_it import MarkdownIt
from pydantic import BaseModel, Extra, PrivateAttr, root_validator, validator
from ray.air import ScalingConfig as AIRScalingConfig
from ray.serve.config import AutoscalingConfig

from aviary.backend.llm.dict_utils import merge_dicts
from aviary.backend.llm.error_handling import TooManyStoppingSequences
from aviary.common.models import ErrorResponse, Message, Prompt, PromptFormat  # noqa
from aviary.env_conf import MAX_NUM_STOPPING_SEQUENCES

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound=BaseModel)


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
            self.dict(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                skip_defaults=skip_defaults,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
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
    generated_text: Optional[str] = None
    num_input_tokens: Optional[int] = None
    num_input_tokens_batch: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    num_generated_tokens_batch: Optional[int] = None
    preprocessing_time: Optional[float] = None
    generation_time: Optional[float] = None
    timestamp: Optional[float] = None
    finish_reason: Optional[str] = None
    error: Optional[ErrorResponse] = None

    @validator("timestamp", always=True)
    def set_timestamp(cls, v):
        return v or time.time()

    @root_validator
    def text_or_error_or_finish_reason(cls, values):
        if (
            values.get("generated_text") is None
            and values.get("error") is None
            and values.get("finish_reason") is None
        ):
            raise ValueError(
                "Either 'generated_text' or 'error' or 'finish_reason' must be set"
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

        generated_text = "".join(
            [response.generated_text or "" for response in responses]
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

        return cls(
            generated_text=generated_text,
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


class S3MirrorConfig(BaseModelExtended):
    bucket_uri: Optional[str] = None
    s3_sync_args: Optional[List[str]] = None
    s3_aws_credentials: Optional[S3AWSCredentials] = None


class GenerationConfig(BaseModelExtended):
    prompt_format: PromptFormat
    generate_kwargs: Dict[str, Any] = {}
    stopping_sequences: Optional[List[Union[str, int, List[Union[str, int]]]]] = None

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


class EngineConfig(BaseModelExtended, extra=Extra.forbid):
    type: str


class SchedulingMetadata(BaseModelExtended):
    request_id: str
    priority: QueuePriority


class SamplingParams(BaseModelExtended):
    """
    Args:
        max_tokens: The maximum number of tokens to generate. Defaults to inf.
        temperature: What sampling temperature to use.
        top_p: An alternative to sampling with temperature, called nucleus sampling.
        n: How many completions to generate for each prompt.
        logprobs: Include the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens.
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
    """

    _ignored_fields: Set[str] = set()

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: int = 1

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
        generate_kwargs = merge_dicts(
            parameters,
            generation.generate_kwargs,
        )

        # The stoppping sequence needs to be merged manually
        generate_kwargs["stop"] = (parameters.get("stop") or []) + (
            generation.stopping_sequences or []
        )

        return cls.parse_obj(generate_kwargs)


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
    messages: List[Message]
    stream: bool = False
    echo: Optional[bool] = False
    user: Optional[str] = None


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
                "engine_config": {
                    "generation",
                    "model_url",
                    "model_description",
                }
            }
        )


class ServeArgs(BaseModel):
    models: Union[str, LLMApp, List[Union[str, LLMApp]]]


class AppArgs(BaseModel):
    model: Union[str, LLMApp]


class RouterArgs(BaseModel):
    models: Dict[str, Union[str, LLMApp]]


class PlacementConfig(BaseModel):
    world_size: int
    scaling_config: ScalingConfig
