from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union

from .error_handling import ErrorReason
from .types import InferenceTask, Request

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )


class AbstractInferenceWorker(ABC):
    @abstractmethod
    def process_new_batch(
        self, requests: List[Request], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        pass

    @abstractmethod
    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        pass

    @abstractmethod
    def filter_tasks(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        pass

    @abstractmethod
    def requires_padding(self) -> bool:
        pass

    @abstractmethod
    def can_infer_max_batch_total_tokens(self) -> bool:
        pass

    def get_inference_task_cls(self) -> Type[InferenceTask]:
        return InferenceTask

    def report_stats(self):  # noqa: B027
        pass


class AsyncInferenceWorker(AbstractInferenceWorker):
    @abstractmethod
    async def process_new_batch_async(
        self, requests: List[Request], batch_id: int
    ) -> Tuple[Union[List["Generation"], ErrorReason], int]:
        pass

    @abstractmethod
    async def generate_next_token_async(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        pass

    @abstractmethod
    async def filter_tasks_async(
        self, batch_id: int, request_ids: List[int]
    ) -> Optional[int]:
        pass
