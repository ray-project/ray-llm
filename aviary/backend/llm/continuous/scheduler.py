import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING, List, Optional, Tuple

from ray._private.utils import run_background_task

from aviary.backend.llm.continuous.policy import TaskSelectionPolicy
from aviary.backend.llm.continuous.tokenstream import FinishReason
from aviary.backend.llm.continuous.types import InferenceTask

from .error_handling import ErrorReason

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )

    from .worker import AsyncInferenceWorker

logger = logging.getLogger(__name__)


_batch_id = -1


def _reset_batch_id():
    global _batch_id
    _batch_id = -1


def get_batch_id() -> int:
    global _batch_id
    _batch_id += 1
    return _batch_id


@dataclass
class Stats:
    num_tasks_processed: int = 0
    num_tasks_failed: int = 0
    num_tasks_pending: int = 0
    num_active_tasks: int = 0
    num_finished_tasks: int = 0
    num_tokens_generated: int = 0
    num_input_tokens: int = 0
    num_iterations: int = 0
    last_report_time: float = 0.0
    start_time: float = 0.0
    first_task_time: float = None

    def report_stats(self):
        new_report_time = time.time()
        if not self.last_report_time:
            self.last_report_time = new_report_time
        if new_report_time - self.last_report_time < 1:
            return False
        elapsed_since_start = self.last_report_time - self.start_time
        elapsed_since_first_task = self.last_report_time - (self.first_task_time or 0)
        token_s = (self.num_input_tokens + self.num_tokens_generated) / (
            new_report_time - self.last_report_time
        )
        logger.info(
            f"scheduler stats: {self}\nelapsed since start: {elapsed_since_start}, elapsed since first task {elapsed_since_first_task}, tokens/s: {token_s} (in last {new_report_time-self.last_report_time}s)"
        )
        self.last_report_time = new_report_time
        return True

    def task_selected(self, tasks: List[InferenceTask]):
        if self.first_task_time is None:
            self.first_task_time = time.time()
        self.num_active_tasks += len(tasks)

    def task_processed(self, tasks: List[InferenceTask]):
        self.num_tasks_processed += len(tasks)
        self.num_input_tokens += sum([t.input_length for t in tasks])

    def task_finished(self):
        self.num_active_tasks -= 1
        self.num_finished_tasks += 1

    def task_failed(self):
        self.num_active_tasks -= 1
        self.num_tasks_failed += 1

    def token_generated(self, num):
        self.num_tokens_generated += num

    def iteration_finished(self):
        self.num_iterations += 1

    def set_num_tasks_pending(self, num):
        self.num_tasks_pending = num

    def start(self):
        self.start_time = time.time()


def _raise_task_exception(
    task: asyncio.Task,
    *,
    scheduler: "InferenceScheduler",
) -> None:
    scheduler.stop()
    try:
        task.result()
    except Exception as e:
        raise RuntimeError("Scheduling loop task finished unexpectedly.") from e
    raise RuntimeError("Scheduling loop task finished unexpectedly.")


class InferenceScheduler:
    """Schedule user requests to be processed by the inference worker and run them potentially asynchronously.

    Args:
        inference_worker: The inference worker to process the tasks.
        task_selection_policy: the scheduling policy to use when selecting tasks
            to run on every iteration.
        task_queue: a queue to add and retrieve tasks from.
        inline: whether to run the scheduling loop in the main thread.

    Attributes:
        add_task: add a task to the scheduler.
        run_tasks: run 1 iteration of the scheduling loop. Note that users should only
            call this if they set inline to True in the constructor. By default it is
            False and the scheduling loop is run in a background thread.
        stop: stop the scheduling loop.
        is_stopped: check if the scheduling loop is stopped.
        check_health: check if the scheduling loop is still running.

    """

    def __init__(
        self,
        inference_worker: "AsyncInferenceWorker",
        task_selection_policy: TaskSelectionPolicy,
        task_queue: asyncio.Queue,
        inline: bool = False,
    ):
        self._task_selection_policy = task_selection_policy
        self._inference_worker = inference_worker
        self._task_queue = task_queue
        self._queue_put_event = asyncio.Event()
        self._lock = Lock()
        self._stop = False
        self._stats = Stats()
        self.scheduling_loop_task = None
        if not inline:
            self.scheduling_loop_task = run_background_task(self._run_scheduling_loop())
            self.scheduling_loop_task.add_done_callback(
                partial(_raise_task_exception, scheduler=self)
            )

    def stop(self):
        """Stop the scheduling loop."""
        with self._lock:
            self._stop = True

    def is_stopped(self) -> bool:
        """Checks if the scheduling loop is stopped."""
        with self._lock:
            return self._stop

    def check_health(self) -> None:
        """Check if the scheduling loop is still running.

        Raises:
            RuntimeError: if the scheduling loop is stopped or dead.
        """
        if self.scheduling_loop_task is not None and self.scheduling_loop_task.done():
            self.stop()
        if self.is_stopped():
            raise RuntimeError("Scheduling loop is stopped or dead.")

    def add_task(self, task: InferenceTask) -> None:
        """Add a task to the scheduler.

        Args:
            task: the task to add.

        """
        self._task_queue.put_nowait(task)
        self._queue_put_event.set()

    async def _run_scheduling_loop(self):
        """Schedule tasks to be processed by the inference worker."""
        # start work the in the scheduling loop to avoid GPU memory leak.
        self._stats.start()

        # The main schedule loop:
        #
        # 0. start with empty in-process tasks.
        #
        # 1. select new tasks to process, based
        # on the current in-process tasks. send them to the inference worker.
        #
        # 2. for both new and in-process tasks, combine them
        # and generate the next token. filter out finished tasks.
        #
        # 3. goto step 1.
        batch_id = None
        in_process_tasks = []
        iterations_since_last_new_batch = 1
        while not self.is_stopped():
            # select new tasks to process.
            (
                in_process_tasks,
                iterations_since_last_new_batch,
                batch_id,
            ) = await self.run_tasks(
                batch_id=batch_id,
                in_process_tasks=in_process_tasks,
                iterations_since_last_new_batch=iterations_since_last_new_batch,
            )

    async def run_tasks(
        self,
        in_process_tasks: List[InferenceTask],
        iterations_since_last_new_batch: int,
        batch_id: Optional[int] = None,
    ):
        """Run 1 iteration of the scheduling loop.

        Args:
            in_process_tasks: the tasks in the current batch that are being processed.
            iterations_since_last_new_batch: the number of iterations since the last
                new batch was processed.
            batch_id: the id of the current batch that is being processed.

        Returns:
            the tasks in the current batch that are being processed after this scheduler
                iteration.
            the number of iterations since the last new batch was processed after this
                scheduler iteration.
            the id of the current batch that is being processed after this scheduler
                iteration.

        """

        new_tasks: List[InferenceTask] = await self._select_new_tasks(
            in_process_tasks,
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )

        if not new_tasks:
            tasks_to_generate_on = in_process_tasks
            tasks_to_generate_on_ids = [batch_id]
            new_unfinished_tasks = []
        else:
            iterations_since_last_new_batch = 0
            # run prefill on the new tasks
            new_batch_id, new_unfinished_tasks = await self._process_new_tasks(
                new_tasks
            )
            # combine new batch with existing batch to generate next token.
            tasks_to_generate_on = in_process_tasks + new_unfinished_tasks
            tasks_to_generate_on_ids = [batch_id, new_batch_id]

        if tasks_to_generate_on:
            iterations_since_last_new_batch += 1

        batch_id, in_process_tasks = await self._generate_next_token(
            tasks=tasks_to_generate_on,
            batch_ids=tasks_to_generate_on_ids,
        )

        if not in_process_tasks:
            iterations_since_last_new_batch = 0

        self._stats.iteration_finished()
        self._report_stats()

        return (
            in_process_tasks,
            iterations_since_last_new_batch,
            batch_id,
        )

    async def _process_new_tasks(
        self, tasks: List[InferenceTask]
    ) -> Tuple[Optional[int], List[InferenceTask]]:
        """Digests new tasks by running prefill on them and loading them into the inference worker.

        Note: Any tasks that are immediatly completed after prefill will be removed
            from the scheduler.

        Args:
            tasks: the new tasks to process.

        Returns:
            The batch id of the new tasks. If batch id is None, then all tasks
                were immediately completed.
            The tasks that are not immediately completed after prefill.
        """
        if len(tasks) == 0:
            return None, []
        batch_id = get_batch_id()
        logger.info(
            f"Processing new batch {batch_id}. Num tasks: {len(tasks)} "
            f"Total tokens: {sum(t.total_tokens for t in tasks)} "
            f"Prefill tokens: {sum(t.input_length for t in tasks)}"
        )
        generations, batch_id = await self._inference_worker.process_new_batch_async(
            requests=[t.request for t in tasks], batch_id=batch_id
        )

        if isinstance(generations, ErrorReason):
            self._handle_errors(generations, tasks)
            return None, []

        self._stats.task_processed(tasks)

        # If after prefill the task is immediately completed, then
        # make them as completed on the inference worker.
        tasks, need_filter = self._process_generation_result(generations, tasks)
        if need_filter and batch_id:
            batch_id = await self._inference_worker.filter_tasks_async(
                batch_id, [t.id for t in tasks]
            )
        return batch_id, tasks

    async def _generate_next_token(
        self, batch_ids: List[int], tasks: List[InferenceTask]
    ) -> Tuple[Optional[int], List[InferenceTask]]:
        """Generate the next token for the given batch_ids.

        Note: This function assumes that first _process_new_tasks is called on the tasks
            so that they are prefilled and loaded on the inference workers first.

        Note: If given multiple batch_ids, the inference worker will roll those batches
            with those ids into a single batch with the id of the newest batch id.

        Args:
            batch_ids: the batch ids to generate the next token for.
            tasks: the tasks that are being processed.

        Returns:
            The batch id of the tasks that are being processed. If batch id is None, then
                all tasks were immediately completed.
            The tasks that are not immediately completed after generating the next token.

        """
        generations, batch_id = await self._inference_worker.generate_next_token_async(
            batch_ids,
        )

        if isinstance(generations, ErrorReason):
            self._handle_errors(generations, tasks)
            return None, []

        tasks, need_filter = self._process_generation_result(generations, tasks)

        if batch_id is not None:
            if need_filter:
                batch_id = await self._inference_worker.filter_tasks_async(
                    batch_id, [t.id for t in tasks]
                )
        else:
            assert len(tasks) == 0, "expect no tasks left"
        return batch_id, tasks

    def _report_stats(self):
        """Report any stats about the scheduler or inference worker."""
        if self._stats.report_stats():
            self._stats.set_num_tasks_pending(self._task_queue.qsize())
            self._inference_worker.report_stats()

    async def _select_new_tasks(
        self,
        in_process_tasks: List[InferenceTask],
        iterations_since_last_new_batch: Optional[int],
    ) -> List[InferenceTask]:
        """Select the next batch of tasks to be run from the queue and in process tasks.

        Args:
            in_process_tasks: the tasks that are currently being processed.
            iterations_since_last_new_batch: the number of iterations since the last new
                batch.

        Tasks are selected based on the self._task_selection_policy.

        Returns:
            The tasks that are selected to be run in the next iteration on the inference
                worker.

        """

        # if there is no in-process tasks and no new tasks in the queue,
        # wait for new tasks to arrive in the queue.
        while (
            len(in_process_tasks) == 0
            and self._task_queue.empty()
            and not self.is_stopped()
        ):
            await self._queue_put_event.wait()
            self._queue_put_event.clear()

        tasks = self._task_selection_policy.select_new_tasks(
            in_process_tasks=in_process_tasks,
            queue=self._task_queue,
            requires_padding=self._inference_worker.requires_padding(),
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )
        self._stats.task_selected(tasks)
        return tasks

    def _process_generation_result(
        self, generations: List["Generation"], tasks: List[InferenceTask]
    ) -> Tuple[List[InferenceTask], bool]:
        """Process the results from each generation/task

        This will do any bookkeeping on the tasks and transfer the generated tokens
            to the output streams of each task.

        Args:
            generations: the tokens that were generated after the most recent iteration
                on the inference worker.
            tasks: the tasks that correspond to those generations.

        Returns:
            Any unfinished tasks.
            Whether or not any tasks were finished.

        """
        some_task_finished = False
        unfinished_tasks = []
        self._stats.token_generated(len(generations))
        # We do not have a guarantee that generations and tasks are in the same order.
        # So we need to match them by task id.
        tasks = {t.id: t for t in tasks}
        for generation in generations:
            task = tasks[generation.request_id]
            # if generation.generated_text.finish_reason == 0 is then the finish reason
            # is due to the generation length, otherwise it's due to EOS/stop token
            generation_finished_eos = (
                generation.generated_text is not None
                and generation.generated_text.finish_reason > 0
            )
            if not generation.token_is_special and not generation_finished_eos:
                # the task is incomplete so the most recent token generated should
                # be written to the task's output stream.
                task.output_stream.put(generation.token_text)
            else:
                # the task is either finished with EOS or a special token was generated.
                task.output_stream.put("")

            # generation.generated_text is only populated if the task is finished.
            generation_finished = generation.generated_text is not None

            # if task.is_finished is True at this point, it means that the task was
            # cancelled. Cancelled means that the task was in the middle of generation,
            # but the user's session disconnected.
            if generation_finished or task.is_finished:
                if generation.generated_text is not None:
                    text = generation.generated_text.text
                else:
                    text = ""
                self._stats.task_finished()
                cancelled = task.is_finished
                if not cancelled:
                    # the task is actually completed
                    task.output_stream.end(
                        FinishReason.from_tgi_finish_reason(
                            generation.generated_text.finish_reason
                        ),
                        generated_text=text,
                    )
                    # task.is_finished now has been set to True
                    assert task.is_finished
                    finish_reason = generation.generated_text.finish_reason
                else:
                    # the task was cancelled
                    finish_reason = task.output_stream.finish_reason
                time_taken = time.time() - task._submit_time_s
                if time_taken:
                    logger.info(
                        f"Task {task.id} (cancelled: {cancelled}) finished in {time_taken}s "
                        f"({task.actual_total_tokens} total tokens, "
                        f"{task.generated_tokens} generated tokens, "
                        f"{task.actual_total_tokens/time_taken} total tokens/s, "
                        f"{task.generated_tokens/time_taken} generated tokens/s) "
                        f"reason: {finish_reason}, response: {generation.generated_text}"
                    )
                some_task_finished = True
                self._task_selection_policy.on_task_finished(task)
            else:
                unfinished_tasks.append(task)
        return unfinished_tasks, some_task_finished

    def _handle_errors(self, error: ErrorReason, tasks: List[InferenceTask]):
        """Handle any errors that occur during inference on the inference worker.

        Args:
            error: the error that occurred.
            tasks: the tasks that were being processed when the error occurred.

        """
        for task in tasks:
            task.mark_as_invalid(error)
            self._stats.task_failed()
