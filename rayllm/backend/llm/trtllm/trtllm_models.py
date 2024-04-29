from typing import Optional

from rayllm.backend.server.models import (
    BaseModelExtended,
    EngineConfig,
    EngineType,
    GenerationConfig,
    LLMApp,
    ModelType,
    S3MirrorConfig,
    SamplingParams,
)
from pydantic import ConfigDict

try:
    from tensorrt_llm.libs import trt_llm_engine_py as trt_py

    class TRTLLMGPTServeConfig(BaseModelExtended):
        engine_dir: str
        model_type: trt_py.TrtGptModelType = None
        scheduler_policy: trt_py.SchedulerPolicy = None
        logger_level: trt_py.LogLevel = None
        max_num_sequences: int = None
        max_tokens_in_paged_kv_cache: int = None
        kv_cache_free_gpu_mem_fraction: float = None
        enable_trt_overlap: bool = None
        model_config = ConfigDict(arbitrary_types_allowed=True)

        @classmethod
        def from_engine_config(
            cls, model_path: str, engine_config: "TRTLLMEngineConfig"
        ):
            params = {}
            if engine_config.scheduler_policy:
                if (
                    engine_config.scheduler_policy
                    == trt_py.SchedulerPolicy.MAX_UTILIZATION.name
                ):
                    params["scheduler_policy"] = trt_py.SchedulerPolicy.MAX_UTILIZATION
                elif (
                    engine_config.scheduler_policy
                    == trt_py.SchedulerPolicy.GUARANTEED_NO_EVICT.name
                ):
                    params[
                        "scheduler_policy"
                    ] = trt_py.SchedulerPolicy.GUARANTEED_NO_EVICT
                else:
                    raise ValueError(
                        f"Unexpected value for trtllm scheduler_policy {engine_config.scheduler_policy}"
                    )

            if engine_config.logger_level:
                if engine_config.logger_level == trt_py.LogLevel.INFO.name:
                    params["logger_level"] = trt_py.LogLevel.INFO
                elif engine_config.logger_level == trt_py.LogLevel.ERROR.name:
                    params["logger_level"] = trt_py.LogLevel.ERROR
                elif engine_config.logger_level == trt_py.LogLevel.INTERNAL_ERROR.name:
                    params["logger_level"] = trt_py.LogLevel.INTERNAL_ERROR
                elif engine_config.logger_level == trt_py.LogLevel.VERBOSE.name:
                    params["logger_level"] = trt_py.LogLevel.VERBOSE
                elif engine_config.logger_level == trt_py.LogLevel.WARNING.name:
                    params["logger_level"] = trt_py.LogLevel.WARNING
                else:
                    raise ValueError(
                        f"Unexpected value for trtllm logger_level {engine_config.logger_level}"
                    )

            if engine_config.model_type:
                if (
                    engine_config.model_type
                    == trt_py.TrtGptModelType.InflightBatching.name
                ):
                    params["model_type"] = trt_py.TrtGptModelType.InflightBatching
                elif (
                    engine_config.model_type
                    == trt_py.TrtGptModelType.InflightFusedBatching.name
                ):
                    params["model_type"] = trt_py.TrtGptModelType.InflightFusedBatching
                else:
                    raise ValueError(
                        f"Unexpected value for trtllm model_type {engine_config.model_type}"
                    )

            if engine_config.max_num_sequences:
                params["max_num_sequences"] = engine_config.max_num_sequences

            if engine_config.max_tokens_in_paged_kv_cache:
                params[
                    "max_tokens_in_paged_kv_cache"
                ] = engine_config.max_tokens_in_paged_kv_cache

            if engine_config.kv_cache_free_gpu_mem_fraction:
                params[
                    "kv_cache_free_gpu_mem_fraction"
                ] = engine_config.kv_cache_free_gpu_mem_fraction

            params["enable_trt_overlap"] = engine_config.enable_trt_overlap

            return cls(
                engine_dir=model_path,
                **params,
            )

except Exception as e:
    exc = e

    class TRTLLMGPTServeConfig(BaseModelExtended):
        def __new__(cls, *args, **kwargs):
            raise exc


class TRTLLMSamplingParams(SamplingParams):
    _ignored_fields = {
        "n",
        "logprobs",
        "stop",
        "best_of",
        "logit_bias",
        "frequency_penalty",
        "response_format",
        "top_logprobs",
    }
    presence_penalty: Optional[float] = None


class TRTLLMGenerationRequest(BaseModelExtended):
    prompt: str
    request_id: str
    sampling_params: TRTLLMSamplingParams
    stream: bool = True


class TRTLLMEngineConfig(EngineConfig):
    type: EngineType = EngineType.TRTLLMEngine
    model_type: ModelType = ModelType.text_generation
    model_id: str
    s3_mirror_config: S3MirrorConfig = None

    # If this is set, engine will try to load model from local path,
    # and inogre the s3_mirror_config.
    model_local_path: str = None
    generation: GenerationConfig

    # TRTLLM GPT server config
    # TODO[Sihan] Set better default values and add comments.
    scheduler_policy: str = "MAX_UTILIZATION"
    logger_level: str = "INFO"
    model_type: str = "InflightFusedBatching"
    max_num_sequences: int = None
    max_tokens_in_paged_kv_cache: int = None
    kv_cache_free_gpu_mem_fraction: float = None
    enable_trt_overlap: bool = True


class TRTLLMApp(LLMApp):
    engine_config: TRTLLMEngineConfig
