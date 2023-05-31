from typing import List, Union

import torch
from transformers import LogitsProcessor, StoppingCriteria

from aviary.backend.logger import get_logger

logger = get_logger(__name__)


class StopOnTokens(StoppingCriteria):
    """
    Stopping criteria to allow stopping on multi-token sequences.

    ``first_stopping_token_in_batch`` attribute can be used for postprocessing after
    generation.

    Args:
        stopping_sequences (List[Union[List[int], int]]): List of sequences to stop on.
    """

    def __init__(self, stopping_sequences: List[Union[List[int], int]]) -> None:
        self.stopping_sequences = stopping_sequences
        self.stop_ids = [
            torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
            for stop_id in self.stopping_sequences
        ]
        self.first_stopping_token_in_batch = {}

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for batch_index, batch in enumerate(input_ids):
            if batch_index not in self.first_stopping_token_in_batch:
                for stop_id in self.stop_ids:
                    if len(batch) > len(stop_id) and batch[-len(stop_id) :].equal(
                        stop_id.to(batch.device)
                    ):
                        self.first_stopping_token_in_batch[batch_index] = len(batch) - 1
                        break
        return len(self.first_stopping_token_in_batch) == len(input_ids)


class StopOnTokensLogitsProcessor(LogitsProcessor):
    """
    Processor to force only EOS token after encountering a stopping sequence.

    Args:
        stopping_sequences (List[Union[List[int], int]]): List of sequences to stop on.
        eos_token_id (Union[int, List[int]]): EOS token id(s).
    """

    def __init__(
        self,
        stopping_sequences: List[Union[List[int], int]],
        eos_token_id: Union[int, List[int]],
    ) -> None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id = eos_token_id
        self.stop_ids = [
            torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
            for stop_id in stopping_sequences
        ]
        self._stopped_batches = set()
        self._nulled_batch = None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for batch_index, batch in enumerate(input_ids):
            if batch_index not in self._stopped_batches:
                for stop_id in self.stop_ids:
                    if len(batch) > len(stop_id) and batch[-len(stop_id) :].equal(
                        stop_id.to(batch.device)
                    ):
                        self._stopped_batches.add(batch_index)
                        break
            if batch_index in self._stopped_batches:
                if self._nulled_batch is None:
                    scores[batch_index, :] = -float("inf")
                    scores[batch_index, self.eos_token_id] = 0
                    self._nulled_batch = scores[batch_index]
                scores[batch_index] = self._nulled_batch
        return scores
