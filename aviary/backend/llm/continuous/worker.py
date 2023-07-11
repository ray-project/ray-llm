from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .error_handling import ErrorReason
from .types import Request as GenerationRequest

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )


class AbstractInferenceWorker(ABC):
    @abstractmethod
    def process_new_batch(
        self, requests: List["GenerationRequest"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        pass

    @abstractmethod
    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        pass

    @abstractmethod
    def filter_requests(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        pass

    @abstractmethod
    def requires_padding(self) -> bool:
        pass

    def report_stats(self):  # noqa: B027
        pass


class AsyncInferenceWorker(AbstractInferenceWorker):
    @abstractmethod
    async def process_new_batch_async(
        self, requests: List["GenerationRequest"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        pass

    @abstractmethod
    async def generate_next_token_async(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        pass

    @abstractmethod
    async def filter_requests_async(
        self, batch_id: int, request_ids: List[int]
    ) -> Optional[int]:
        pass
