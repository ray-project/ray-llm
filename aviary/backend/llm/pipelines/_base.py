from abc import ABC, abstractmethod
from queue import Queue
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

import torch

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Prompt, Response

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class BasePipeline(ABC):
    def __init__(
        self,
        model,
        tokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        self.model = model
        self.prompt_format: str = prompt_format or ""
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def __call__(
        self,
        inputs: List[Union[str, Prompt]],
        **kwargs,
    ) -> List[Response]:
        raise NotImplementedError()

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "BasePipeline":
        model, tokenizer = initializer.load(model_id)
        logger.info(f"Model: {model}")
        return cls(
            model,
            tokenizer,
            prompt_format=prompt_format,
            device=device,
            **kwargs,
        )


class StreamingPipeline(BasePipeline):
    def stream(
        self,
        inputs: List[Union[str, Prompt]],
        queue: Queue,
        **kwargs,
    ) -> Iterator[List[Response]]:
        raise NotImplementedError()


class AsyncStreamingPipeline(BasePipeline):
    async def async_stream(
        self,
        inputs: List[Union[str, Prompt]],
        queue: Queue,
        **kwargs,
    ):
        raise NotImplementedError()
