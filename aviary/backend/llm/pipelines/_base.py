from abc import ABC, abstractmethod
from queue import Queue
from typing import TYPE_CHECKING, Iterator, List, Optional, Union

import torch

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Response

if TYPE_CHECKING:
    from ..initializers._base import LLMInitializer

logger = get_logger(__name__)


class BasePipeline(ABC):
    def __init__(
        self,
        model,
        tokenizer,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def __call__(
        self,
        inputs: List[str],
        **kwargs,
    ) -> List[Response]:
        raise NotImplementedError()

    @classmethod
    def from_initializer(
        cls,
        initializer: "LLMInitializer",
        model_id: str,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "BasePipeline":
        model, tokenizer = initializer.load(model_id)
        logger.info(f"Model: {model}")
        return cls(
            model,
            tokenizer,
            device=device,
            **kwargs,
        )


class StreamingPipeline(BasePipeline):
    def stream(
        self,
        inputs: List[str],
        queue: Queue,
        **kwargs,
    ) -> Iterator[List[Response]]:
        raise NotImplementedError()


class AsyncStreamingPipeline(BasePipeline):
    async def async_stream(
        self,
        inputs: List[str],
        queue: Queue,
        **kwargs,
    ):
        raise NotImplementedError()
