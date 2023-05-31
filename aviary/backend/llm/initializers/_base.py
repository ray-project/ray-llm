from abc import ABC, abstractmethod
from typing import Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aviary.backend.logger import get_logger

logger = get_logger(__name__)


class LLMInitializer(ABC):
    """Initialize model and tokenizer and place them on the correct device.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ):
        self.device = device
        self.world_size = world_size

    def load(self, model_id: str) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        """Load model and tokenizer.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model = self.load_model(model_id)
        tokenizer = self.load_tokenizer(model_id)
        return self.postprocess(model, tokenizer)

    @abstractmethod
    def load_model(self, model_id: str) -> "PreTrainedModel":
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        pass

    @abstractmethod
    def load_tokenizer(self, tokenizer_id: str) -> "PreTrainedTokenizer":
        """Load tokenizer.

        Args:
            tokenizer_id (str): Hugging Face tokenizer name.
        """
        pass

    def postprocess(
        self, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"
    ) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        """Postprocess model and tokenizer.

        Args:
            model (PreTrainedModel): Model to postprocess.
            tokenizer (PreTrainedTokenizer): Tokenizer to postprocess.
        """
        return self.postprocess_model(model), self.postprocess_tokenizer(tokenizer)

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Postprocess model.

        Args:
            model (PreTrainedModel): Model to postprocess.
        """
        return model

    def postprocess_tokenizer(
        self, tokenizer: "PreTrainedTokenizer"
    ) -> "PreTrainedTokenizer":
        """Postprocess tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to postprocess.
        """
        return tokenizer
