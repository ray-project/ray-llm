from typing import List

import torch
from llama_cpp import LogitsProcessor, StoppingCriteria
from transformers import MaxTimeCriteria, MinNewTokensLengthLogitsProcessor

from aviary.backend.logger import get_logger

logger = get_logger(__name__)


class LlamaCppMinNewTokensLengthLogitsProcessor(
    MinNewTokensLengthLogitsProcessor, LogitsProcessor
):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        scores = MinNewTokensLengthLogitsProcessor.__call__(
            self, torch.LongTensor(input_ids), torch.FloatTensor(scores)[None, :]
        )
        return scores[0].tolist()


class LlamaMaxTimeCriteria(MaxTimeCriteria, StoppingCriteria):
    pass
