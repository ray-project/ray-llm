import logging
from enum import Enum
from typing import Optional

from rayllm.backend.server.models import (
    Embeddings,
    EngineConfig,
    EngineType,
    GCSMirrorConfig,
    LLMApp,
    ModelType,
    S3MirrorConfig,
)

logger = logging.getLogger(__name__)


class EmbeddingOptimize(str, Enum):
    ONNX = "onnx"


class EmbeddingEngineConfig(EngineConfig):
    class Config:
        use_enum_values = True

    type: EngineType = EngineType.EmbeddingEngine
    model_type: ModelType = ModelType.embedding
    model_url: Optional[str] = None
    model_description: Optional[str] = None
    optimize: Optional[EmbeddingOptimize] = None

    s3_mirror_config: Optional[S3MirrorConfig] = None
    gcs_mirror_config: Optional[GCSMirrorConfig] = None

    max_total_tokens: int = 512
    max_batch_size: int = 1
    batch_wait_timeout_s: float = 0.1

    @property
    def embeddings_model(self):
        return Embeddings


class EmbeddingApp(LLMApp):
    engine_config: EmbeddingEngineConfig  # type: ignore
