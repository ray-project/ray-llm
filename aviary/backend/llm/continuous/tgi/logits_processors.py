from typing import List, Union

import torch
from transformers import LogitsProcessor, MinNewTokensLengthLogitsProcessor


class MinNewTokensLogitsProcessor(MinNewTokensLengthLogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.

    Args:
        generated_tokens (`int`):
            Tokens generated so far.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int]]`):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(
        self,
        generated_tokens: int,
        min_new_tokens: int,
        eos_token_id: Union[int, List[int]],
    ):
        for arg_name, arg_value in [
            ("generated_tokens", generated_tokens),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(
                    f"`{arg_name}` has to be a positive integer, but is {arg_value}"
                )

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        self.generated_tokens = generated_tokens
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self.generated_tokens += 1
        if self.generated_tokens <= self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        return scores


def batched_bincount(
    x: torch.Tensor, dim: torch.Tensor, max_value: int
) -> torch.Tensor:
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


class HeterogeneousFrequencyPresencePenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing an additive penalty on repeated sequences, with
    separate penalty for presence and frequence (see https://platform.openai.com/docs/api-reference/parameter-details).
    This version allows for a separate value for each sample and runs inplace when possible.
    It doesn't validate inputs.

    Args:
        presence_penalty (`List[float]`):
            The parameter for presence penalty. 0.0 means no penalty.
        frequency_penalty (`List[float]`):
            The parameter for frequence penalty. 0.0 means no penalty.
    """

    def __init__(
        self,
        presence_penalty: List[float],
        frequency_penalty: List[float],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty_tensor = torch.tensor(
            presence_penalty, dtype=dtype, device=device
        ).unsqueeze(1)
        self.frequency_penalty_tensor = torch.tensor(
            frequency_penalty, dtype=dtype, device=device
        ).unsqueeze(1)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        occurences = batched_bincount(input_ids, 1, scores.shape[1])
        frequency_penalty = occurences * self.frequency_penalty_tensor
        presence_penalty = (occurences > 0) * self.presence_penalty_tensor

        scores.sub_(frequency_penalty).sub_(presence_penalty)
        return scores

    def filter(self, indices):
        self.presence_penalty = [self.presence_penalty[i] for i in indices]
        self.frequency_penalty = [self.frequency_penalty[i] for i in indices]
        if any([x != 0.0 for x in self.presence_penalty]) or any(
            [x != 0.0 for x in self.frequency_penalty]
        ):
            self.presence_penalty_tensor = self.presence_penalty_tensor[indices]
            self.frequency_penalty_tensor = self.frequency_penalty_tensor[indices]
            return self
        return None
