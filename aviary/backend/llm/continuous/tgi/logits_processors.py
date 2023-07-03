from typing import List, Union

import torch
from transformers import MinNewTokensLengthLogitsProcessor


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
        if self.generated_tokens < self.min_new_tokens:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")

        return scores
