from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aviary.backend.logger import get_logger

from .hf_transformers import TransformersInitializer

try:
    from aviary.backend.llm.continuous.tgi.tgi_worker import TGIInferenceWorker
except ImportError as e:
    TGIInferenceWorker = e

logger = get_logger(__name__)


class TextGenerationInferenceInitializer(TransformersInitializer):
    """Initialize model (using text-generation-inference).

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        **from_pretrained_kwargs,
    ):
        if isinstance(TGIInferenceWorker, Exception):
            raise RuntimeError(
                "TextGenerationInferenceInitializer requires text-generation-inference to be installed."
            ) from TGIInferenceWorker

        self.device = device
        self.dtype = dtype
        self.from_pretrained_kwargs = from_pretrained_kwargs

    def load_model(self, model_id: str) -> "TGIInferenceWorker":
        model_id_or_location = self._get_model_location_on_disk(model_id)
        if model_id != model_id_or_location:
            safetensor_files = list(Path(model_id_or_location).glob("*.safetensors"))
            if safetensor_files:
                model_id = model_id_or_location
        logger.info(f"Loading model from {model_id}")
        return TGIInferenceWorker(
            model_id,
            dtype=str(self.dtype).replace("torch.", ""),
            **self._get_model_from_pretrained_kwargs(),
        )

    def load_tokenizer(self, tokenizer_id: str) -> "PreTrainedTokenizer":
        # Initialized inside TGIInferenceWorker
        return None

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        return model
