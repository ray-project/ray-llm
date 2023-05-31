import os
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch
from huggingface_hub import hf_hub_download

from aviary.backend.logger import get_logger

from ._base import LLMInitializer

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = get_logger(__name__)


class LlamaCppTokenizer:
    """Thin wrapper around a llama_cpp model to provide a subset of the PreTrainedTokenizer interface"""

    def __init__(self, model: "Llama") -> None:
        self.model = model

    def decode(self, tokens: Union[List[int], List[List[int]]], **kwargs) -> str:
        if not tokens:
            return tokens
        if isinstance(tokens[0], int):
            return self.model.detokenize(tokens).decode("utf-8")
        return [self.decode(t) for t in tokens]

    def encode(self, text: Union[str, List[str], List[List[str]]], **kwargs) -> str:
        if isinstance(text, str):
            return self.model.tokenize(text.encode("utf-8"))
        return [self.encode(t) for t in text]

    def batch_encode(self, text: Union[List[str], List[List[str]]], **kwargs) -> str:
        return self.encode(text)

    def __call__(self, text: Union[str, List[str], List[List[str]]], **kwargs):
        return self.encode(text, **kwargs)


class LlamaCppInitializer(LLMInitializer):
    """Initialize llama_cpp model and tokenizer.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        model_filename (str): Name of the model file to download from HuggingFace Hub.
            This needs to be in the ``model_id`` repository (passed to ``self.load()``).
        **model_init_kwargs: Keyword arguments to pass to the llama_cpp model init.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        model_filename: str,
        **model_init_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
        )
        self.model_filename = model_filename
        self.model_init_kwargs = model_init_kwargs

    def _get_model_init_kwargs(self) -> Dict[str, Any]:
        return {
            # We use a large integer to put all of the layers on GPU by default.
            "n_gpu_layers": 0 if self.device.type == "cpu" else 10**6,
            "seed": 0,
            "verbose": False,
            "n_threads": int(os.environ["OMP_NUM_THREADS"]),
            **self.model_init_kwargs,
        }

    def load_model(self, model_id: str) -> "Llama":
        model_path = hf_hub_download(model_id, self.model_filename)

        # TODO upstream. Lazy import to avoid issues on CPU head node
        from aviary.backend.llm.initializers._llama_impl import LlamaWithMinLen

        return LlamaWithMinLen(
            model_path=os.path.abspath(model_path),
            **self._get_model_init_kwargs(),
        )

    def load_tokenizer(self, tokenizer_name: str) -> None:
        return None

    def postprocess(
        self, model: "Llama", tokenizer: None
    ) -> Tuple["Llama", LlamaCppTokenizer]:
        return super().postprocess(model, LlamaCppTokenizer(model))
