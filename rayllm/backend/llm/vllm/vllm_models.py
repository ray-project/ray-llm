import logging
from typing import Any, Dict, Optional

from rayllm.backend.server.models import (
    EngineConfig,
    EngineType,
    GenerationConfig,
    GenerationRequest,
    LLMApp,
    ModelType,
    SamplingParams,
)

logger = logging.getLogger(__name__)


class VLLMEngineConfig(EngineConfig):
    type: EngineType = EngineType.VLLMEngine
    model_type: ModelType = ModelType.text_generation
    model_url: Optional[str] = None
    model_description: Optional[str] = None
    generation: GenerationConfig

    engine_kwargs: Dict[str, Any] = {}

    max_total_tokens: int = 2048

    @property
    def sampling_params_model(self):
        return VLLMSamplingParams


class VLLMSamplingParams(SamplingParams):
    """
    Args:
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        seed: Seed for deterministic sampling with temperature>0.
    """

    _ignored_fields = {"best_of", "n", "logit_bias"}

    top_k: Optional[int] = None
    seed: Optional[int] = None


class VLLMGenerationRequest(GenerationRequest):
    sampling_params: VLLMSamplingParams


class VLLMApp(LLMApp):
    engine_config: VLLMEngineConfig  # type: ignore


class FunctionCallingVLLMApp(VLLMApp):
    depends_on_model: Optional[str]
    standalone_function_calling_model: Optional[bool]
