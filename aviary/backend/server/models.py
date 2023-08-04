import time
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from huggingface_hub import hf_hub_download, hf_hub_url
from markdown_it import MarkdownIt
from pydantic import BaseModel, Extra, root_validator, validator
from ray.air import ScalingConfig as AIRScalingConfig
from ray.serve.config import AutoscalingConfig

from aviary.common.models import ErrorResponse, Prompt, PromptFormat  # noqa


def markdown_extract_first_paragraph(markdown_text: str):
    """Extract the first paragraph from a markdown-formatted string."""
    from mdit_py_plugins.front_matter import front_matter_plugin

    md = MarkdownIt("commonmark", {"breaks": True, "html": True}).use(
        front_matter_plugin
    )
    tokens = md.parse(markdown_text)
    first_paragraph = []
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
    def parse_yaml(cls, file, **kwargs) -> "BaseModelExtended":
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
        return super().dict(*args, **kwargs)

    def json(
        self,
        *args,
        **kwargs,
    ) -> str:
        self.__dict__.update(
            {prop: getattr(self, prop) for prop in self.get_properties()}
        )

        return super().json(*args, **kwargs)


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
        num_input_tokens = max(num_input_tokens) if num_input_tokens else None
        num_input_tokens_batch = [
            response.num_input_tokens_batch
            for response in responses
            if response.num_input_tokens_batch is not None
        ]
        num_input_tokens_batch = (
            max(num_input_tokens_batch) if num_input_tokens_batch else None
        )
        num_generated_tokens = [
            response.num_generated_tokens
            for response in responses
            if response.num_generated_tokens is not None
        ]
        num_generated_tokens = (
            sum(num_generated_tokens) if num_generated_tokens else None
        )
        num_generated_tokens_batch = [
            response.num_generated_tokens_batch
            for response in responses
            if response.num_generated_tokens_batch is not None
        ]
        num_generated_tokens_batch = (
            sum(num_generated_tokens_batch) if num_generated_tokens_batch else None
        )
        preprocessing_time = [
            response.preprocessing_time
            for response in responses
            if response.preprocessing_time is not None
        ]
        preprocessing_time = max(preprocessing_time) if preprocessing_time else None
        generation_time = [
            response.generation_time
            for response in responses
            if response.generation_time is not None
        ]
        generation_time = sum(generation_time) if generation_time else None
        error = next(
            (response.error for response in reversed(responses) if response.error), None
        )

        return cls(
            generated_text=generated_text,
            num_input_tokens=num_input_tokens,
            num_input_tokens_batch=num_input_tokens_batch,
            num_generated_tokens=num_generated_tokens,
            num_generated_tokens_batch=num_generated_tokens_batch,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
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
            return self.num_input_tokens + self.num_generated_tokens
        except Exception:
            return None

    @property
    def num_total_tokens_batch(self) -> Optional[float]:
        try:
            return self.num_input_tokens_batch + self.num_generated_tokens_batch
        except Exception:
            return None

    @property
    def total_time_per_token(self) -> Optional[float]:
        try:
            return self.total_time / self.num_total_tokens
        except Exception:
            return None

    @property
    def generation_time_per_token(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_total_tokens
        except Exception:
            return None

    @property
    def total_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.total_time / self.num_total_tokens_batch
        except Exception:
            return None

    @property
    def generation_time_per_token_batch(self) -> Optional[float]:
        try:
            return self.generation_time / self.num_total_tokens_batch
        except Exception:
            return None


class SchedulerPolicyConfig(BaseModelExtended, extra=Extra.forbid):
    type: str


class QuotaBasedTaskSelectionPolicyConfig(SchedulerPolicyConfig):
    type: Literal["QuotaBasedTaskSelectionPolicy"] = "QuotaBasedTaskSelectionPolicy"
    # Max total tokens (input+output) in a batch of multiple tasks
    max_batch_total_tokens: Optional[int] = None
    # Max total tokens (input+output) in a single task. Shouldn't be higher than
    # context length of the model.
    max_total_tokens: int = 2048
    # This setting defines how many tokens can be passed before forcing the waiting
    # queries to be put on the batch (if the size of the batch allows for it).
    # tgi calls this max_waiting tokens
    max_iterations_curr_batch: int = 20
    # Max input tokens in a single task. Note: this is not validated yet
    max_input_length: int = 1024
    # Limits the number of tokens for the prefill operation.
    max_batch_prefill_tokens: int = 4096
    # This represents the ratio of waiting queries vs running queries where
    # you want to start considering pausing the running queries to include the waiting
    # ones into the same batch.
    waiting_served_ratio: float = 1.2

    @root_validator
    def validate_values(cls, values):
        if values.get("max_input_length") > values.get("max_batch_prefill_tokens"):
            raise ValueError(
                f"max_batch_prefill_tokens ({values.get('max_batch_prefill_tokens')}) must be >= max_input_length ({values.get('max_input_length')})"
            )
        if values.get("max_batch_total_tokens"):
            if values.get("max_batch_prefill_tokens") > values.get(
                "max_batch_total_tokens"
            ):
                raise ValueError(
                    f"max_batch_prefill_tokens ({values.get('max_batch_prefill_tokens')}) must be <= max_batch_total_tokens ({values.get('max_batch_total_tokens')})"
                )
            if values.get("max_total_tokens") > values.get("max_batch_total_tokens"):
                raise ValueError(
                    f"max_total_tokens ({values.get('max_total_tokens')}) must be <= max_batch_total_tokens ({values.get('max_batch_total_tokens')})"
                )
        return values


class SchedulerConfig(BaseModelExtended, extra=Extra.forbid):
    policy: Union[SchedulerPolicyConfig, QuotaBasedTaskSelectionPolicyConfig]


class S3MirrorConfig(BaseModelExtended):
    bucket_uri: Optional[str] = None
    s3_sync_args: Optional[List[str]] = None


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


class TextGenerationInferenceEngineConfig(BaseModelExtended):
    type: str
    model_id: str
    model_url: Optional[str] = None
    model_description: Optional[str] = None
    generation: GenerationConfig
    scheduler: SchedulerConfig

    s3_mirror_config: Optional[S3MirrorConfig] = None
    runtime_env: Optional[Dict[str, Any]] = None
    hf_model_id: Optional[str] = None
    model_init_kwargs: Dict[str, Any] = {}

    @root_validator(pre=True)
    def resolve_model_url_and_description(cls, values):
        model_id = values.get("model_id")
        model_url = values.get("model_url")
        model_description = values.get("model_description")
        if not model_url:
            # If we do not have a model URL, use model ID to
            # get it from HF Hub
            model_url = hf_hub_url(model_id, "dummy")
            model_url = model_url[: model_url.rfind("/resolve")]
            values["model_url"] = model_url
        if not model_description:
            # If we do not have a model description, use model ID to
            # obtain it from HF Hub and get the first text paragraph
            # from readme. This is not foolproof, but should work
            # OK for most cases.
            try:
                readme = hf_hub_download(model_id, "README.md")
                assert readme
                with open(readme, "r") as f:
                    model_description = markdown_extract_first_paragraph(f.read())
            except Exception:
                model_description = ""
            values["model_description"] = model_description
        return values

    @property
    def actual_hf_model_id(self) -> str:
        return self.hf_model_id or self.model_id

    def get_initialization_kwargs(self) -> dict:
        """
        Get kwargs that will be actually passed to the LLMInitializer
        constructor.
        """
        return self.model_init_kwargs.copy()


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
    engine_config: TextGenerationInferenceEngineConfig
    scaling_config: ScalingConfig

    @property
    def air_scaling_config(self) -> AIRScalingConfig:
        return self.scaling_config.as_air_scaling_config()


class DeploymentConfig(BaseModelExtended):
    autoscaling_config: Optional[AutoscalingConfig]
    max_concurrent_queries: Optional[int] = None
    ray_actor_options: Optional[Dict[str, Any]] = None


class LLMApp(Args):
    """The full configuration of a single LLM Model"""

    deployment_config: Optional[DeploymentConfig] = None
    enabled: bool = True


class ServeArgs(BaseModel):
    models: Union[str, LLMApp, List[Union[str, LLMApp]]]


class AppArgs(BaseModel):
    model: Union[str, LLMApp]


class RouterArgs(BaseModel):
    models: Dict[str, Union[str, LLMApp]]
