import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from aviary.backend.logger import get_logger

from .._base import LLMInitializer

logger = get_logger(__name__)


class TransformersInitializer(LLMInitializer):
    """Initialize model and tokenizer and place them on the correct device.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        self.device = device
        self.world_size = world_size
        self.dtype = dtype
        self.from_pretrained_kwargs = from_pretrained_kwargs
        self.use_bettertransformer = use_bettertransformer
        self.torch_compile = torch_compile

    def _get_model_from_pretrained_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for AutoModel.from_pretrained."""
        return self.from_pretrained_kwargs

    def load(self, model_id: str) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
        """Load model and tokenizer.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model = self.load_model(model_id)
        tokenizer = self.load_tokenizer(model_id)
        return self.postprocess_model(model), self.postprocess_tokenizer(tokenizer)

    def _get_model_location_on_disk(self, model_id: str) -> str:
        """Get the location of the model on disk.

        Args:
            model_id (str): Hugging Face model ID.
        """
        from transformers.utils.hub import TRANSFORMERS_CACHE

        path = os.path.expanduser(
            os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
        )
        model_id_or_path = model_id

        if os.path.exists(path):
            with open(os.path.join(path, "refs", "main"), "r") as f:
                snapshot_hash = f.read().strip()
            if os.path.exists(os.path.join(path, "snapshots", snapshot_hash)):
                model_id_or_path = os.path.join(path, "snapshots", snapshot_hash)
        return model_id_or_path

    def load_model(self, model_id: str) -> "PreTrainedModel":
        """Load model.

        Args:
            model_id (str): Hugging Face model ID.
        """
        model_id_or_path = self._get_model_location_on_disk(model_id)

        logger.info(f"Loading model {model_id_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, **self._get_model_from_pretrained_kwargs()
        )
        model.eval()
        return model

    def load_tokenizer(self, tokenizer_id: str) -> "PreTrainedTokenizer":
        """Load tokenizer.

        Args:
            tokenizer_id (str): Hugging Face tokenizer name.
        """
        tokenizer_id_or_path = self._get_model_location_on_disk(tokenizer_id)

        # TODO make this more robust, add logging
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_id_or_path, padding_side="left", trust_remote_code=True
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                tokenizer_id, padding_side="left", trust_remote_code=True
            )

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        """Postprocess model.

        First, transform the model with BetterTransformer if use_bettertransformer is True.
        Then, compile the model with torch.compile() if torch_compile is not None, using
        the provided parameters.

        Args:
            model (PreTrainedModel): Model to postprocess.
        """
        if self.use_bettertransformer:
            from optimum.bettertransformer import BetterTransformer

            logger.info("Transforming the model with BetterTransformer...")
            model = BetterTransformer.transform(model)

        if self.torch_compile and self.torch_compile["backend"]:
            logger.info("Compiling the model with torch.compile()...")
            model = torch.compile(model, **self.torch_compile)

        return model

    def postprocess_tokenizer(
        self, tokenizer: "PreTrainedTokenizer"
    ) -> "PreTrainedTokenizer":
        """Postprocess tokenizer.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to postprocess.
        """
        return tokenizer


class DeviceMapInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device(s).

    Uses Hugging Face Transformer's ``device_map`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for torch.compile. Defaults to None.
        device_map (str, optional): Device map to use (same as in AutoModel.from_pretrained). Defaults to "auto".
        **from_pretrained_kwargs: Keyword arguments for AutoModel.from_pretrained.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )
        self.device_map = device_map

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            device_map=self.device_map,
            **self.from_pretrained_kwargs,
        )


class SingleDeviceInitializer(TransformersInitializer):
    """Initialize model and tokenizer and place them on the correct device.

    Uses Hugging Face Transformer's ``device`` argument.

    Args:
        device (torch.device): Device to place model and tokenizer on.
        world_size (int): Number of GPUs to use.
        dtype (torch.dtype, optional): Data type to use. Defaults to torch.float16.
        use_bettertransformer (bool, optional): Whether to use BetterTransformer. Defaults to False.
        torch_compile (Optional[Dict[str, Any]], optional): Parameters for ``torch.compile``. Defaults to None.
        **from_pretrained_kwargs: Keyword arguments for ``AutoModel.from_pretrained``.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        dtype: torch.dtype = torch.float16,
        use_bettertransformer: bool = False,
        torch_compile: Optional[Dict[str, Any]] = None,
        **from_pretrained_kwargs,
    ):
        super().__init__(
            device=device,
            world_size=world_size,
            dtype=dtype,
            use_bettertransformer=use_bettertransformer,
            torch_compile=torch_compile,
            **from_pretrained_kwargs,
        )

    def _get_model_from_pretrained_kwargs(self):
        return dict(
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            **self.from_pretrained_kwargs,
        )

    def postprocess_model(self, model: "PreTrainedModel") -> "PreTrainedModel":
        return super().postprocess_model(model.to(device=self.device))
