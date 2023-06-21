import warnings
from typing import Any, Dict, List, Literal, Optional, Set, Union

import torch
import yaml
from huggingface_hub import hf_hub_download, hf_hub_url
from markdown_it import MarkdownIt
from mdit_py_plugins.front_matter import front_matter_plugin
from pydantic import BaseModel, Extra, Field, root_validator, validator
from ray.air import ScalingConfig as AIRScalingConfig
from ray.serve.config import AutoscalingConfig
from typing_extensions import Annotated


def markdown_extract_first_paragraph(markdown_text: str):
    """Extract the first paragraph from a markdown-formatted string."""
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


class Prompt(BaseModelExtended):
    prompt: str
    use_prompt_format: bool = True

    def __str__(self) -> str:
        return self.prompt


class ErrorResponse(BaseModelExtended):
    error: str
    error_type: str


class Response(ComputedPropertyMixin, BaseModelExtended):
    generated_text: str
    num_input_tokens: Optional[int] = None
    num_input_tokens_batch: Optional[int] = None
    num_generated_tokens: Optional[int] = None
    num_generated_tokens_batch: Optional[int] = None
    preprocessing_time: Optional[float] = None
    generation_time: Optional[float] = None

    @classmethod
    def merge_stream(cls, *responses: "Response") -> "Response":
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

        return cls(
            generated_text=generated_text,
            num_input_tokens=num_input_tokens,
            num_input_tokens_batch=num_input_tokens_batch,
            num_generated_tokens=num_generated_tokens,
            num_generated_tokens_batch=num_generated_tokens_batch,
            preprocessing_time=preprocessing_time,
            generation_time=generation_time,
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

    def __str__(self) -> str:
        return self.generated_text


class TorchCompile(BaseModelExtended):
    backend: Optional[str] = "inductor"
    mode: Optional[str] = None
    fullgraph: bool = False
    dynamic: bool = False
    options: Optional[Dict[str, Any]] = None


class Initializer(BaseModelExtended, extra=Extra.forbid):
    type: str

    @root_validator(pre=True)
    def set_type(cls, values):
        values["type"] = cls.__name__
        return values

    def get_initializer_kwargs(self) -> dict:
        """
        Get kwargs that will be actually passed to the LLMInitializer
        constructor.
        """
        return self.dict(exclude={"type"})

    @property
    def allowed_pipelines(self) -> Set[str]:
        return {}


class Transformers(Initializer, extra=Extra.forbid):
    use_bettertransformer: bool = False
    torch_compile: Optional[TorchCompile] = None
    dtype: Union[
        Literal["float16"], Literal["bfloat16"], Literal["float32"], Literal["int8"]
    ] = "float16"
    from_pretrained_kwargs: Dict[str, Any] = {}

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def get_initializer_kwargs(self) -> dict:
        return {
            **self.dict(exclude={"type", "from_pretrained_kwargs", "dtype"}),
            "dtype": self.torch_dtype,
            **self.from_pretrained_kwargs,
        }

    @property
    def allowed_pipelines(self) -> Set[str]:
        return {"transformers"}


class DeepSpeed(Transformers):
    type: Literal["DeepSpeed"]
    dtype: Union[Literal["float16"], Literal["float32"], Literal["int8"]] = "float16"
    use_kernel: bool = False
    max_tokens: int = 1024
    use_meta_tensor: bool = False
    ds_inference_kwargs: Optional[Dict[str, Any]] = None

    @root_validator
    def use_kernel_bettertransformer_torch_compile(cls, values):
        if values.get("use_kernel") and (
            values.get("use_bettertransformer") or values.get("torch_compile")
        ):
            raise ValueError(
                "Cannot combine 'use_bettertransformer' or 'torch_compile' with 'use_kernel=True'."
            )
        return values

    @root_validator
    def use_kernel_use_meta_tensor(cls, values):
        if not values.get("use_kernel") and values.get("use_meta_tensor"):
            raise ValueError("'use_meta_tensor=True' needs 'use_kernel=True'.")
        return values


class DeviceMap(Transformers):
    type: Literal["DeviceMap"]
    device_map: Optional[str] = "auto"


class SingleDevice(Transformers):
    type: Literal["SingleDevice"]


class LlamaCpp(Initializer):
    type: Literal["LlamaCpp"]
    model_filename: str
    model_init_kwargs: Dict[str, Any] = {}

    def get_initializer_kwargs(self) -> dict:
        return {
            **self.dict(exclude={"type", "model_init_kwargs"}),
            **self.model_init_kwargs,
        }

    @property
    def allowed_pipelines(self) -> Set[str]:
        return {"llamacpp"}


class S3MirrorConfig(BaseModelExtended):
    bucket_uri: Optional[str] = None
    s3_sync_args: Optional[List[str]] = None


class InitializationConfig(BaseModelExtended):
    initializer: Annotated[
        Union[DeepSpeed, DeviceMap, SingleDevice, LlamaCpp], Field(discriminator="type")
    ]
    pipeline: Union[Literal["transformers"], Literal["llamacpp"], Literal["default"]]
    s3_mirror_config: Optional[S3MirrorConfig] = None
    runtime_env: Optional[Dict[str, Any]] = None
    hf_model_id: Optional[str] = None
    full_warmup: bool = False  # For debugging purposes

    @root_validator
    def initializer_pipeline(cls, values):
        pipeline = values.get("pipeline")
        if pipeline == "default":
            warnings.warn(
                "'default' pipeline is deprecated. Use 'transformers' instead. This will raise an error in the future.",
                DeprecationWarning,
                stacklevel=2,
            )
            pipeline = "transformers"
            values["pipeline"] = pipeline
        initializer: Initializer = values.get("initializer")
        if pipeline and pipeline not in initializer.allowed_pipelines:
            raise ValueError(
                f"'{pipeline}' pipeline cannot be used with '{initializer.type}' initializer. "
                f"Allowed pipelines for this initializer are {initializer.allowed_pipelines}."
            )
        return values


class GenerationConfig(BaseModelExtended):
    prompt_format: Optional[str] = None
    max_batch_size: int = 1
    batch_wait_timeout_s: int = 1
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": 256,
        "do_sample": True,
        "top_p": 0.92,
        "top_k": 0,
    }
    stopping_sequences: Optional[List[Union[str, int, List[Union[str, int]]]]] = None

    @validator("prompt_format")
    def check_prompt_format(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "prompt_format must be None, empty string or string containing '{instruction}'"
        return value

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


class LLMConfig(BaseModelExtended):
    model_id: str
    initialization: InitializationConfig
    generation: GenerationConfig
    model_url: Optional[str] = None
    model_description: Optional[str] = None
    # TODO make this token-based
    max_input_words: int = 400

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
        return self.initialization.hf_model_id or self.model_id


class ScalingConfig(BaseModelExtended):
    num_workers: int
    num_gpus_per_worker: float = 1
    num_cpus_per_worker: float = 1
    placement_strategy: str = "PACK"
    resources_per_worker: Optional[Dict[str, float]] = None
    pg_timeout_s: float = 600

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
    model_config: LLMConfig
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
