import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import ray
from ray._private.utils import run_background_task

from aviary.backend.llm.continuous.policy import QuotaBasedRequestSelectionPolicy
from aviary.backend.llm.continuous.queue import InferenceRequest
from aviary.backend.llm.continuous.tokenstream import TokenStream

from .error_handling import ErrorReason, OutOfMemory
from .types import Request

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )

    from .worker import AbstractInferenceWorker, AsyncInferenceWorker

logger = logging.getLogger(__name__)

_request_id = -1

_batch_id = -1


def _reset_batch_id():
    global _batch_id
    _batch_id = -1


def get_batch_id() -> int:
    global _batch_id
    _batch_id += 1
    return _batch_id


def get_request_id() -> int:
    # TODO: more robust request id generation.
    global _request_id
    _request_id += 1
    return _request_id


class Tokenizer(ABC):
    @abstractmethod
    def get_input_length(self, input_text: str) -> int:
        raise NotImplementedError("")


class NaiveTokenizer(Tokenizer):
    def get_input_length(self, input_text: str) -> int:
        return min(input_text.count(" ") + 1)

    # TODO: add model specific tokenizer


class TransformersTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def get_input_length(self, input_text: str) -> int:
        return self._tokenizer(
            text=input_text,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
        )["input_ids"].shape[1]


class RayTokenizer(Tokenizer):
    def __init__(self, worker_group: List[ray.ObjectRef]):
        self._worker_group = worker_group
        self._id = -1

    def get_input_length(self, input_text: str) -> ray.ObjectRef:
        self._id += 1
        # Simple round robin
        return self._worker_group[
            self._id % len(self._worker_group)
        ].get_input_length.remote(input_text)


@dataclass
class Stats:
    num_requests_processed: int = 0
    num_requests_failed: int = 0
    num_requests_pending: int = 0
    num_active_requests: int = 0
    num_finished_requests: int = 0
    num_tokens_generated: int = 0
    num_input_tokens: int = 0
    num_iterations: int = 0
    last_report_time: float = 0.0
    start_time: float = 0.0
    first_request_time: float = None

    def report_stats(self):
        if time.time() - self.last_report_time < 1:
            return False
        self.last_report_time = time.time()
        elapsed_since_start = self.last_report_time - self.start_time
        elapsed_since_first_request = self.last_report_time - (
            self.first_request_time or 0
        )
        token_s = (
            self.num_input_tokens + self.num_tokens_generated
        ) / elapsed_since_first_request
        logger.info(
            f"scheduler stats: {self}\nelapsed since start: {elapsed_since_start}, elapsed since first request {elapsed_since_first_request}, tokens/s: {token_s}"
        )
        return True

    def request_selected(self, requests: List[InferenceRequest]):
        if self.first_request_time is None:
            self.first_request_time = time.time()
        self.num_active_requests += len(requests)

    def request_processed(self, requests: List[InferenceRequest]):
        self.num_requests_processed += len(requests)
        self.num_input_tokens += sum([r.input_length for r in requests])

    def request_finished(self):
        self.num_active_requests -= 1
        self.num_finished_requests += 1

    def request_failed(self):
        self.num_active_requests -= 1
        self.num_requests_failed += 1

    def token_generated(self, num):
        self.num_tokens_generated += num

    def iteration_finished(self):
        self.num_iterations += 1

    def set_num_requests_pending(self, num):
        self.num_requests_pending = num

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


def _asyncio_queue_put_nowait_left(queue: asyncio.Queue, item: Any) -> None:
    if queue.full():
        raise asyncio.queues.QueueFull
    queue._queue.appendleft(item)
    queue._unfinished_tasks += 1
    queue._finished.clear()
    queue._wakeup_next(queue._getters)


# TODO make this non-TGI specific
class InferenceScheduler:
    def __init__(
        self,
        tokenizer: Tokenizer,
        inference_worker_loader: Callable[[], "AbstractInferenceWorker"],
        request_selection_policy: QuotaBasedRequestSelectionPolicy,  # RequestSelectionPolicy,
        request_queue: asyncio.Queue,
        inline: bool = False,
    ):
        self._tokenizer = tokenizer
        self._request_selection_policy = request_selection_policy
        self._inference_worker_loader = inference_worker_loader
        self._request_queue = request_queue
        self._queue_put_event = asyncio.Event()
        self._lock = Lock()
        self._stop = False
        self._stats = Stats()
        self._has_oom = False
        self.scheduling_loop_task = None
        if not inline:
            self.scheduling_loop_task = run_background_task(self._run_scheduling_loop())
            self.scheduling_loop_task.add_done_callback(
                partial(_raise_task_exception, scheduler=self)
            )

    def stop(self):
        with self._lock:
            self._stop = True

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def check_health(self) -> None:
        if self.scheduling_loop_task is not None and self.scheduling_loop_task.done():
            self.stop()
        if self.is_stopped():
            raise RuntimeError("Scheduling loop is stopped or dead.")

    def process_request(
        self,
        input_text: str,
        params: Dict[str, Any],
        max_new_tokens: int = 256,
        max_length: int = 1024,
    ) -> TokenStream:
        request = Request(
            id=get_request_id(),
            inputs=input_text,
            truncate=max_length,
            max_new_tokens=max_new_tokens,
            params=params,
        )
        return self._add_request(request)

    def _add_request(self, request: Request) -> TokenStream:
        pending_request = InferenceRequest.from_request(
            request,
            request_input_length=self._tokenizer.get_input_length(request.inputs),
        )
        self._request_queue.put_nowait(pending_request)
        self._queue_put_event.set()
        return pending_request.output_stream

    async def _run_scheduling_loop(self):
        """Schedule requests to be processed by the inference worker."""
        # start work the in the scheduling loop to avoid GPU memory leak.
        self._inference_worker: "AbstractInferenceWorker" = (
            self._inference_worker_loader()
        )
        self._stats.start()

        # The main schedule loop:
        #
        # 0. start with empty in-process requests.
        #
        # 1. select new requests to process, based
        # on the current in-process requests. send them to the inference worker.
        #
        # 2. for both new and in-process requests, combine them
        # and generate the next token. filter out finished requests.
        #
        # 3. goto step 1.
        batch_id = None
        in_process_requests = []
        await asyncio.sleep(0.000001)
        while not self.is_stopped():
            # select new requests to process.
            new_requests = await self._select_new_requests(in_process_requests)
            new_batch_id, new_unfinished_requests = self._process_new_requests(
                new_requests
            )

            # combine new batch with existing batch to generate next token.
            batch_id, in_process_requests = self._generate_next_token(
                [batch_id, new_batch_id],
                in_process_requests + new_unfinished_requests,
            )

            self._stats.iteration_finished()
            self._report_stats()
            await asyncio.sleep(0.000001)

    def _report_stats(self):
        if self._stats.report_stats():
            self._stats.set_num_requests_pending(self._request_queue.qsize())
            self._inference_worker.report_stats()

    async def _select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
        iterations_since_last_new_batch: Optional[int],
    ) -> List[InferenceRequest]:
        while (
            len(in_process_requests) == 0
            and self._request_queue.empty()
            and not self.is_stopped()
        ):
            # if there is no in-process requests and no new requests in the queue,
            # wait for new requests to arrive in the queue.
            # self._request_queue.wait(1)

            await self._queue_put_event.wait()
            self._queue_put_event.clear()

        requests = self._request_selection_policy.select_new_requests(
            in_process_requests,
            self._request_queue,
            has_oom=self._has_oom,
            requires_padding=self._inference_worker.requires_padding(),
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )
        self._has_oom = False
        self._stats.request_selected(requests)
        return requests

    def _process_new_requests(
        self, requests: List[InferenceRequest]
    ) -> Tuple[Optional[int], List[InferenceRequest]]:
        if len(requests) == 0:
            return None, []
        batch_id = get_batch_id()
        logger.info(
            f"Processing new batch {batch_id}. Num requests: {len(requests)} Total tokens: {sum(r.total_tokens for r in requests)} Prefill tokens: {sum(r.request_input_length for r in requests)}"
        )
        generations, batch_id = self._inference_worker.process_new_batch(
            [r.request for r in requests], batch_id=batch_id
        )

        # handle ooms
        if isinstance(generations, OutOfMemory):
            logger.info(f"OOM detected in new batch {generations}.")
            return self._handle_ooms(batch_id, requests)
        elif isinstance(generations, ErrorReason):
            self._handle_errors(generations, requests)
            return None, []

        self._stats.request_processed(requests)

        requests, need_filter = self._process_generation_result(generations, requests)

        if need_filter and batch_id:
            batch_id = self._inference_worker.filter_requests(
                batch_id, [r.id for r in requests]
            )
        return batch_id, requests

    def _generate_next_token(
        self, batch_ids: List[int], requests: List[InferenceRequest]
    ) -> Tuple[Optional[int], List[InferenceRequest]]:
        generations, batch_id = self._inference_worker.generate_next_token(
            batch_ids,
        )

        # handle ooms
        if isinstance(generations, OutOfMemory):
            return self._handle_ooms(batch_id, requests)
        elif isinstance(generations, ErrorReason):
            self._handle_errors(generations, requests)
            return None, []

        requests, need_filter = self._process_generation_result(generations, requests)

        if batch_id is not None:
            if need_filter:
                batch_id = self._inference_worker.filter_requests(
                    batch_id, [r.id for r in requests]
                )
        else:
            assert len(requests) == 0, "expect no requests left"
        return batch_id, requests

    def _process_generation_result(
        self, generations: List["Generation"], requests: List[InferenceRequest]
    ) -> Tuple[List[InferenceRequest], bool]:
        some_request_finished = False
        unfinished_requests = []
        self._stats.token_generated(len(generations))
        assert len(generations) == len(
            requests
        ), "expect same number of generations as requests"
        # We do not have a guarantee that generations and requests are in the same order.
        # So we need to match them by request id.
        requests = {r.id: r for r in requests}
        for generation in generations:
            request = requests[generation.request_id]
            # generation.generated_text.finish_reason == 0 is length, otherwise it's due to EOS/stop token
            if not generation.token_is_special and not (
                generation.generated_text is not None
                and generation.generated_text.finish_reason > 0
            ):
                request.output_stream.put(generation.token_text)
            if generation.generated_text is not None or request.is_finished:
                if generation.generated_text is not None:
                    text = generation.generated_text.text
                else:
                    text = ""
                self._stats.request_finished()
                logger.info(
                    f"Request {request.id} (cancelled: {request.is_finished}) finished, response: {generation.generated_text}"
                )
                request.output_stream.end(text)
                some_request_finished = True
                self._request_selection_policy.request_finished(request)
            else:
                unfinished_requests.append(request)
        return unfinished_requests, some_request_finished

    def _handle_recoverable_ooms(self, batch_id, requests: List[InferenceRequest]):
        # pop last request to reduce memory overhead.
        assert requests
        failed_request = requests.pop()
        _asyncio_queue_put_nowait_left(self._request_queue, failed_request)
        self._stats.request_failed()
        batch_id = self._inference_worker.filter_requests(
            batch_id, [r.id for r in requests]
        )
        self._has_oom = True
        return batch_id, requests

    def _handle_ooms(self, batch_id: int, requests: List[InferenceRequest]):
        logger.warning("OOM detected, trying to recover...")
        if batch_id:
            return self._handle_recoverable_ooms(batch_id, requests)

        # oom is not recoverable
        logger.error("OOM not recoverable!")
        while requests:
            failed_request = requests.pop()
            _asyncio_queue_put_nowait_left(self._request_queue, failed_request)
            self._stats.request_failed()
        self._has_oom = True
        return None, []

    def _handle_errors(self, error: ErrorReason, requests: List[InferenceRequest]):
        for request in requests:
            request.mark_as_invalid(error)
            self._stats.request_failed()


class AsyncInferenceScheduler(InferenceScheduler):
    """Same as InferenceScheduler, but _run_scheduling_loop is fully async."""

    async def _run_scheduling_loop(self):
        """Schedule requests to be processed by the inference worker."""
        # start work the in the scheduling loop to avoid GPU memory leak.
        self._inference_worker: "AsyncInferenceWorker" = self._inference_worker_loader()
        self._stats.start()

        # The main schedule loop:
        #
        # 0. start with empty in-process requests.
        #
        # 1. select new requests to process, based
        # on the current in-process requests. send them to the inference worker.
        #
        # 2. for both new and in-process requests, combine them
        # and generate the next token. filter out finished requests.
        #
        # 3. goto step 1.
        batch_id = None
        in_process_requests = []
        iterations_since_last_new_batch = None
        while not self.is_stopped():
            # select new requests to process.
            new_requests = await self._select_new_requests(
                in_process_requests,
                iterations_since_last_new_batch=iterations_since_last_new_batch,
            )

            if new_requests:
                iterations_since_last_new_batch = 0
            elif iterations_since_last_new_batch is not None:
                iterations_since_last_new_batch += 1

            (
                new_batch_id,
                new_unfinished_requests,
            ) = await self._process_new_requests(new_requests)

            # combine new batch with existing batch to generate next token.
            batch_id, in_process_requests = await self._generate_next_token(
                [batch_id, new_batch_id],
                in_process_requests + new_unfinished_requests,
            )

            self._stats.iteration_finished()
            self._report_stats()

    async def _process_new_requests(
        self, requests: List[InferenceRequest]
    ) -> Tuple[Optional[int], List[InferenceRequest]]:
        if len(requests) == 0:
            return None, []
        batch_id = get_batch_id()
        logger.info(
            f"Processing new batch {batch_id}. Num requests: {len(requests)} Total tokens: {sum(r.total_tokens for r in requests)} Prefill tokens: {sum(r.request_input_length for r in requests)}"
        )
        generations, batch_id = await self._inference_worker.process_new_batch_async(
            [r.request for r in requests], batch_id=batch_id
        )

        # handle ooms
        if isinstance(generations, OutOfMemory):
            logger.info(f"OOM detected in new batch {generations}.")
            return self._handle_ooms(batch_id, requests)
        elif isinstance(generations, ErrorReason):
            self._handle_errors(generations, requests)
            return None, []

        self._stats.request_processed(requests)

        requests, need_filter = self._process_generation_result(generations, requests)

        if need_filter and batch_id:
            batch_id = await self._inference_worker.filter_requests_async(
                batch_id, [r.id for r in requests]
            )
        return batch_id, requests

    async def _generate_next_token(
        self, batch_ids: List[int], requests: List[InferenceRequest]
    ) -> Tuple[Optional[int], List[InferenceRequest]]:
        generations, batch_id = await self._inference_worker.generate_next_token_async(
            batch_ids,
        )

        # handle ooms
        if isinstance(generations, OutOfMemory):
            return self._handle_ooms(batch_id, requests)
        elif isinstance(generations, ErrorReason):
            self._handle_errors(generations, requests)
            return None, []

        requests, need_filter = self._process_generation_result(generations, requests)

        if batch_id is not None:
            if need_filter:
                batch_id = await self._inference_worker.filter_requests_async(
                    batch_id, [r.id for r in requests]
                )
        else:
            assert len(requests) == 0, "expect no requests left"
        return batch_id, requests
