from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

T = TypeVar("T")


class SchedulerScalingRequest(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NOOP = "noop"


@dataclass
class EngineStats:
    num_tasks_processed: int = 0
    num_tasks_failed: int = 0
    num_tasks_pending: int = 0
    num_active_tasks: int = 0
    num_finished_tasks: int = 0
    num_tokens_generated: int = 0
    num_input_tokens: int = 0
    num_iterations: int = 0
    utilization: float = 0.0
    scaling_request: SchedulerScalingRequest = SchedulerScalingRequest.NOOP
