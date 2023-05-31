import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Prompt, Response

from ..initializers.llamacpp import LlamaCppInitializer, LlamaCppTokenizer
from ._base import BasePipeline
from .utils import construct_prompts, decode_stopping_sequences_where_needed

if TYPE_CHECKING:
    from llama_cpp import Llama

logger = get_logger(__name__)


class LlamaCppPipeline(BasePipeline):
    """Text generation pipeline using llama.cpp.

    May not support all features."""

    def __init__(
        self,
        model: "Llama",
        tokenizer: LlamaCppTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> None:
        from llama_cpp import Llama

        if not isinstance(model, Llama):
            raise TypeError("Model must be an instance of llama_cpp.Llama.")
        self.model = model
        self.prompt_format: str = prompt_format or ""
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.device = device

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Dict[str, Any]:
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("echo", False)
        stopping_sequences = generate_kwargs.pop("stopping_sequences")
        stopping_sequences = decode_stopping_sequences_where_needed(
            self.tokenizer, stopping_sequences
        )
        generate_kwargs.setdefault("stop", stopping_sequences)
        return generate_kwargs

    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        kwargs = self._add_default_generate_kwargs(kwargs)
        inputs = construct_prompts(inputs, prompt_format=self.prompt_format)

        responses = []
        for input in inputs:
            st = time.monotonic()
            output = self.model(input, **kwargs)
            gen_time = time.monotonic() - st
            text = output["choices"][0]["text"].replace("\u200b", "").strip()
            responses.append(
                Response(
                    generated_text=text,
                    num_generated_tokens=output["usage"]["completion_tokens"],
                    num_input_tokens=output["usage"]["prompt_tokens"],
                    num_generated_tokens_batch=output["usage"]["completion_tokens"],
                    num_input_tokens_batch=output["usage"]["prompt_tokens"],
                    preprocessing_time=0,
                    postprocessing_time=0,
                    generation_time=gen_time,
                )
            )
        return responses

    @classmethod
    def from_initializer(
        cls,
        initializer: "LlamaCppInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "BasePipeline":
        assert isinstance(initializer, LlamaCppInitializer)
        model, tokenizer = initializer.load(model_id)
        logger.info(f"Model: {model}")
        return cls(
            model,
            tokenizer,
            prompt_format=prompt_format,
            device=device,
            **kwargs,
        )

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
