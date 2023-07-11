import asyncio
import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Optional, Tuple

from aviary.backend.llm.continuous.error_handling import InputTooLong
from aviary.backend.llm.continuous.queue import (
    InferenceRequest,
    RequestQueue,
)

logger = logging.getLogger(__name__)


class RequestSelectionPolicy(ABC):
    @abstractmethod
    def select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
        queue: RequestQueue,
        iterations_since_last_new_batch: Optional[int],
        has_oom: bool = False,
        requires_padding: bool = False,
    ) -> Tuple[List[InferenceRequest]]:
        raise NotImplementedError

    @abstractmethod
    def request_finished(self, finished_request: InferenceRequest):
        # noqa
        pass

    # TODO: we might also interested in other events, such as when a request is
    # finished, or when a token is generated.


Quota = namedtuple(
    "Quota", ["min_num_requests", "token_budget", "prefill_token_budget"]
)


class QuotaBasedRequestSelectionPolicy(RequestSelectionPolicy):
    def __init__(
        self,
        max_batch_total_tokens: int = 32000,
        max_batch_prefill_tokens: int = 4096,
        max_input_length: int = 1024,
        waiting_served_ratio: float = 1.2,
        max_waiting_tokens: int = 20,
    ):
        self.max_batch_total_tokens = max_batch_total_tokens
        self.max_batch_prefill_tokens = max_batch_prefill_tokens
        self.max_input_length = max_input_length
        self.waiting_served_ratio = waiting_served_ratio
        self.max_waiting_tokens = max_waiting_tokens
        self.waiting_tokens = 0
        self.oom_penalty = 1.0
        self.oomed_requests = set()

    def request_finished(self, finished_request: InferenceRequest):
        if finished_request.id in self.oomed_requests:
            self.oomed_requests.remove(finished_request.id)
        # if len(self.oomed_requests) == 0:
        #    self.oom_penalty = 1

    def validate_request(self, request: InferenceRequest) -> bool:
        if request.is_finished:
            return False
        if request.request_input_length > self.max_input_length:
            logging.info(f"Request {request.id} is over the max input length.")
            request.mark_as_invalid(
                InputTooLong(request.request_input_length, self.max_input_length)
            )
            return False
        return True

    def _calculate_budget(
        self,
        in_process: List[InferenceRequest],
        selected: List[InferenceRequest],
        candidate: InferenceRequest,
    ):
        max_input_length = candidate.input_length
        gen_length = candidate.gen_length
        for r in in_process:
            max_input_length = max(max_input_length, r.input_length)
            gen_length += r.gen_length
        for r in selected:
            max_input_length = max(max_input_length, r.input_length)
            gen_length += r.gen_length
        return gen_length + max_input_length * (len(selected) + 1 + len(in_process))

    def select_new_requests(
        self,
        in_process_requests: List[InferenceRequest],
        queue: asyncio.Queue,
        iterations_since_last_new_batch: Optional[int],
        has_oom: bool = False,
        requires_padding: bool = False,
    ) -> Tuple[List[InferenceRequest]]:
        if has_oom:
            self.oom_penalty -= 0.05
            if self.oom_penalty < 0.1:
                raise ValueError(
                    "OOM penalty is too low. This suggests an irrecoverable error."
                )
            for r in in_process_requests:
                self.oomed_requests.add(r.id)
            logger.info(f"OOM penalty: {self.oom_penalty}")

        min_num_requests, token_budget, prefill_token_budget = self.calculate_quota(
            in_process_requests,
            requires_padding=requires_padding,
            iterations_since_last_new_batch=iterations_since_last_new_batch,
        )
        org_token_budget = token_budget
        org_prefill_token_budget = prefill_token_budget
        prefill_tokens = 0
        decode_tokens = 0

        if min_num_requests and queue.qsize() < min_num_requests:
            return []

        hypothetical_results: List[InferenceRequest] = []
        request_validity: List[bool] = []
        any_request_is_valid = False
        while len(request_validity) < queue.qsize():
            request: InferenceRequest = queue._queue[len(request_validity)]
            if not self.validate_request(request):
                request_validity.append(False)
                continue

            if requires_padding:
                max_input_length = max(
                    r.request_input_length for r in (hypothetical_results + [request])
                )
                prefill_tokens = (len(hypothetical_results) + 1) * max_input_length
            else:
                prefill_tokens += request.request_input_length

            decode_tokens += request.request.max_new_tokens

            if (
                prefill_tokens > prefill_token_budget
                or (prefill_tokens + decode_tokens) > token_budget
            ):
                break
            any_request_is_valid = True
            request_validity.append(True)
            hypothetical_results.append(request)

        results = []
        if min_num_requests and len(hypothetical_results) < min_num_requests:
            results = []
        else:
            results = []
            if any_request_is_valid:
                logger.info(
                    f"Creating new batch. In process requests: {len(in_process_requests)} Token budget: {org_token_budget} Prefill token budget: {org_prefill_token_budget} Min num requests: {min_num_requests}"
                )
            for is_request_valid in request_validity:
                if is_request_valid:
                    results.append(queue.get_nowait())
                else:
                    # Discard invalid results
                    queue.get_nowait()

        return results

    def calculate_quota(
        self,
        in_process_requests: List[InferenceRequest],
        iterations_since_last_new_batch: Optional[int],
        requires_padding: bool = False,
    ) -> Quota:
        if (
            iterations_since_last_new_batch is None
            or iterations_since_last_new_batch > self.max_waiting_tokens
        ):
            min_num_requests = None
        else:
            min_num_requests = int(len(in_process_requests) * self.waiting_served_ratio)

        return Quota(
            min_num_requests=min_num_requests,
            token_budget=int(self.max_batch_total_tokens * self.oom_penalty)
            - sum(r.total_tokens for r in in_process_requests),
            prefill_token_budget=int(self.max_batch_prefill_tokens * self.oom_penalty),
        )
