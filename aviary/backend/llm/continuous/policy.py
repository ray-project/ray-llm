import asyncio
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional

from aviary.backend.llm.continuous.types import InferenceTask

logger = logging.getLogger(__name__)


class TaskSelectionPolicy(ABC):
    @abstractmethod
    def select_new_tasks(
        self,
        in_process_tasks: List[InferenceTask],
        queue: asyncio.Queue,
        iterations_since_last_new_batch: Optional[int],
        requires_padding: bool = False,
    ) -> List[InferenceTask]:
        """Select new tasks to be added to the batch that is run on the inference workers.

        Args:
            in_process_tasks: Tasks that are in the current batch that is being processed.
            queue: The queue of pending tasks that are waiting to be processed.
            iterations_since_last_new_batch: The number of iterations since the inference
                workers began processing a batch with new tasks.
            requires_padding: Whether the model requires that rows in a batch have the
                same length.

        Returns:
            A list of tasks that should be added to the batch that inference is being
                run on.
        """
        raise NotImplementedError

    @abstractmethod
    def on_task_finished(self, finished_tasks: InferenceTask):
        pass


Quota = namedtuple("Quota", ["token_budget", "prefill_token_budget"])


class QuotaBasedTaskSelectionPolicy(TaskSelectionPolicy):
    """Decides the tasks in the batch that is run on inference workers using bin-packing.

    Args:
        max_batch_total_tokens: The maximum number of tokens that can be placed in a batch
            that is run on the inference workers.
        max_batch_prefill_tokens: The maximum number of tokens in a batch that are prefill
            tokens. Prefill tokens that are generated from the model from the prompt in a
            task.
        max_input_length: The max prefill tokens allowed for a single task.
        max_total_tokens: The maximum number of tokens (prefill + generated) allowed for
            a single task.
        max_iterations_curr_batch: The maximum number of iterations that a batch can be
            processed before a new batch of tasks is processed.
        waiting_served_ratio: The ratio of pending tasks that should be prefilled to
            the number of tasks that are in the batch that is being processed IFF
            the number of iterations since the last new batch was processed is more than
            max_iterations_curr_batch.

    """

    def __init__(
        self,
        max_batch_total_tokens: int = 32000,
        max_batch_prefill_tokens: int = 4096,
        max_input_length: int = 1024,
        max_total_tokens: int = 2048,
        waiting_served_ratio: float = 1.2,
        max_iterations_curr_batch: int = 20,
    ):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.max_batch_prefill_tokens = max_batch_prefill_tokens
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens
        self.waiting_served_ratio = waiting_served_ratio
        self.max_iterations_curr_batch = max_iterations_curr_batch

    def on_task_finished(self, finished_tasks: InferenceTask):
        pass

    def validate_task(self, task: InferenceTask) -> bool:
        """Validate whether task should be added to the next batch that is run."""
        if task.is_finished:
            return False
        return True

    def select_new_tasks(
        self,
        in_process_tasks: List[InferenceTask],
        queue: asyncio.Queue,
        iterations_since_last_new_batch: int,
        requires_padding: bool = False,
    ) -> List[InferenceTask]:
        num_pending_tasks_to_prefill = self._get_num_pending_tasks_to_prefill(
            num_in_process_tasks=len(in_process_tasks),
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )

        token_budget, prefill_token_budget = self.calculate_quota(
            in_process_tasks,
        )

        if queue.qsize() < num_pending_tasks_to_prefill:
            return []

        prefill_tokens = 0
        total_tokens = 0

        task_batch_to_run: List[InferenceTask] = []
        is_task_valid: List[bool] = []
        while len(is_task_valid) < queue.qsize():
            task: InferenceTask = queue._queue[len(is_task_valid)]
            if not self.validate_task(task):
                is_task_valid.append(False)
                continue

            if requires_padding:
                max_input_length = max(
                    r.input_length for r in (task_batch_to_run + [task])
                )
                # + 1 for the task that is currently being evaluated if it is
                # going to be added to the task_batch_to_run.
                prefill_tokens = (len(task_batch_to_run) + 1) * max_input_length
            else:
                prefill_tokens += task.input_cost

            total_tokens += task.total_cost

            if prefill_tokens > prefill_token_budget or total_tokens > token_budget:
                break
            is_task_valid.append(True)
            task_batch_to_run.append(task)

        new_tasks = []

        # if the number of tasks that are going to be placed in the batch is less than
        # the number of pending tasks that should be prefilled, if we were to run
        # the batch then we would be underutilizing our resources. In this case we
        # should yield to the scheduler to do prefill on some of the pending tasks
        # and process a new batch that is more full.
        if len(task_batch_to_run) < num_pending_tasks_to_prefill:
            new_tasks = []
        else:
            new_tasks = []
            if task_batch_to_run:
                logger.info(
                    f"Creating new batch. In process tasks: {len(in_process_tasks)} "
                    f"Token budget: {token_budget} "
                    f"Prefill token budget: {prefill_token_budget} "
                    f"Min. number of pending tasks to do prefill on: {num_pending_tasks_to_prefill}"
                )
            for valid in is_task_valid:
                if valid:
                    new_tasks.append(queue.get_nowait())
                else:
                    queue.get_nowait()

        return new_tasks

    def calculate_quota(
        self,
        in_process_tasks: List[InferenceTask],
    ) -> Quota:
        """Calculate the current budget for scheduling new tasks.

        Calculate the remaining amount of tokens that can be used for new tasks.
        The initial budget corresponds to the amount of tokens that can be
        placed in a batch that is run through the inference workers.

        Args:
            in_process_tasks: Tasks that are currently in process. These are
                used for calculating the remaining token budget.
        Returns:
            A Quota containing the remaining token budgets.
        """

        quota = Quota(
            token_budget=int(self.max_batch_total_tokens)
            - sum([r.total_cost for r in in_process_tasks]),
            prefill_token_budget=int(self.max_batch_prefill_tokens),
        )
        if quota.token_budget < 0:
            raise ValueError("Token budget cannot be negative.")
        return quota

    def _get_num_pending_tasks_to_prefill(
        self, num_in_process_tasks: int, iterations_since_last_new_batch: int
    ):
        """Compute the number of pending tasks that should be prefilled.

        If the number of iterations since the last new batch was processed is greater
        than what is allowed, then this implies that pending requests have been waiting
        too long to be served.

        In this case we should maybe interrupt processing the current batch and do
        prefill for some of the pending requests.

        Args:
            num_in_process_tasks: The number of tasks that are currently in process.
            iterations_since_last_new_batch: The number of iterations since new requests
                were added to the running batch that is being computed by the engine.

        """
        if iterations_since_last_new_batch > self.max_iterations_curr_batch:
            num_pending_tasks_to_prefill = 0
        else:
            num_pending_tasks_to_prefill = int(
                num_in_process_tasks * self.waiting_served_ratio
            )
        return num_pending_tasks_to_prefill
