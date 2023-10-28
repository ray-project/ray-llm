from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union

from rayllm.backend.llm.engine.stats import EngineStats
from rayllm.backend.llm.generation import GenerationStream
from rayllm.backend.server.models import (
    EngineConfig,
    PlacementConfig,
    SamplingParams,
    SchedulingMetadata,
)


class BaseEngine(ABC):
    @abstractmethod
    def __init__(self, engine_config: EngineConfig, placement_config: PlacementConfig):
        ...

    @abstractproperty
    def is_running(self) -> bool:
        ...

    @abstractmethod
    async def start(self):
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: Union[str, List[int]],
        sampling_params: SamplingParams,
        scheduling_metadata: SchedulingMetadata,
    ) -> GenerationStream:
        ...

    # cancellation is in GenerationStream

    @abstractmethod
    def check_health(self) -> bool:
        ...

    @abstractmethod
    def stats(self) -> EngineStats:
        ...

    @abstractmethod
    def shutdown(self, shutdown_pg: bool = True):
        ...
