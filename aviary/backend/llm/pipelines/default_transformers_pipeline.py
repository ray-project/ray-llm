from typing import TYPE_CHECKING, List, Optional, Union

import torch
from transformers import Pipeline as TransformersPipeline
from transformers import PreTrainedModel, PreTrainedTokenizer, pipeline

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Prompt, Response

from ._base import BasePipeline
from .utils import construct_prompts

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class DefaultTransformersPipeline(BasePipeline):
    """Text generation pipeline using Transformers Pipeline.

    May not support all features.

    Args:
        model (PreTrainedModel): Hugging Face model.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        prompt_format (Optional[str], optional): Prompt format. Defaults to None.
        device (Optional[Union[str, int, torch.device]], optional): Device to place model on. Defaults to model's
            device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        if not hasattr(model, "generate"):
            raise ValueError("Model must have a generate method.")
        super().__init__(model, tokenizer, prompt_format, device)

        self.pipeline = None

    def _get_transformers_pipeline(self, **kwargs) -> TransformersPipeline:
        default_kwargs = dict(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=None,
        )
        transformers_pipe = pipeline(**{**default_kwargs, **kwargs})
        transformers_pipe.device = self.device
        return transformers_pipe

    @torch.inference_mode()
    def __call__(self, inputs: List[Union[str, Prompt]], **kwargs) -> List[Response]:
        if not self.pipeline:
            self.pipeline = self._get_transformers_pipeline()
        kwargs = self._add_default_generate_kwargs(kwargs)
        inputs = construct_prompts(inputs, prompt_format=self.prompt_format)
        logger.info(f"Pipeline params: {kwargs}")
        return [
            Response(generated_text=text) for text in self.pipeline(inputs, **kwargs)
        ]

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        stopping_sequences: List[Union[int, str]] = None,
        **kwargs,
    ) -> "DefaultTransformersPipeline":
        default_kwargs = dict(
            model=model_id,
            device=None,
        )
        transformers_pipe = pipeline(
            **{**default_kwargs, **kwargs},
            model_kwargs=initializer.get_model_init_kwargs(),
        )
        transformers_pipe.model = initializer.postprocess_model(transformers_pipe.model)
        pipe = cls(
            model=transformers_pipe.model,
            tokenizer=transformers_pipe.tokenizer,
            prompt_format=prompt_format,
            device=device,
            stopping_sequences=stopping_sequences,
            **kwargs,
        )
        pipe.pipeline = transformers_pipe
        transformers_pipe.device = pipe.device
        return pipe

    def preprocess(self, prompts: List[str], **generate_kwargs):
        pass

    def forward(self, model_inputs, **generate_kwargs):
        pass
