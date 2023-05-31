import torch

from aviary.backend.llm.pipelines.processors import StopOnTokens
from aviary.backend.llm.pipelines.utils import (
    construct_prompts,
    truncate_to_first_stop_token,
)
from aviary.backend.server.models import Prompt


def test_truncate_to_first_stop_token():
    """Test that EOS sequences are correctly truncated from the model output."""
    tokens_list = [1, 1, 1, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(tokens)

    tokens_list = [1, 1, 1, 2]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1, 1, 1])
    )

    tokens_list = [1, 1, 2, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1, 1])
    )

    tokens_list = [2, 1, 2, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([])
    )

    tokens_list = [1, 1, 3, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1, 1, 3, 1])
    )

    tokens_list = [1, 1, 1, 4]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1, 1, 1, 4])
    )

    tokens_list = [1, 1, 3, 4]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1, 1])
    )

    tokens_list = [1, 3, 4, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([1])
    )

    tokens_list = [3, 4, 1, 1]
    tokens = torch.as_tensor(tokens_list)
    assert truncate_to_first_stop_token(tokens, stop_ids=[2, [3, 4]]).equal(
        torch.as_tensor([])
    )


def test_stop_on_tokens_processor():
    """Test that the StopOnTokens processor works with batches and sequences of tokens."""

    # Should not stop here
    processor = StopOnTokens(stopping_sequences=[2, [3, 4]])
    tokens = torch.as_tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    assert not processor(tokens, None)

    for stop_sequence in ([1, 2], [3, 4]):
        # Should not stop here
        processor = StopOnTokens(stopping_sequences=[2, [3, 4]])
        tokens = torch.as_tensor([[1, 1] + stop_sequence, [1, 1, 1, 1]])
        assert not processor(tokens, None)

        # Should not stop here
        processor = StopOnTokens(stopping_sequences=[2, [3, 4]])
        tokens = torch.as_tensor([[1, 1, 1, 1], [1, 1] + stop_sequence])
        assert not processor(tokens, None)

        # Should stop here
        processor = StopOnTokens(stopping_sequences=[2, [3, 4]])
        tokens = torch.as_tensor([[1, 1] + stop_sequence, [1, 1] + stop_sequence])
        assert processor(tokens, None)

        # Should stop here
        processor = StopOnTokens(stopping_sequences=[2, [3, 4]])
        tokens = torch.as_tensor([[1, 1] + stop_sequence, [1, 1, 1, 1]])
        assert not processor(tokens, None)
        tokens = torch.as_tensor([[1, 1] + stop_sequence + [1], [1, 1, 1, 1, 1]])
        assert not processor(tokens, None)
        tokens = torch.as_tensor(
            [[1, 1] + stop_sequence + [1, 1], [1, 1, 1, 1] + stop_sequence]
        )
        assert processor(tokens, None)


def test_construct_prompt():
    """Test that prompts are constructed correctly."""
    text = "Write a short story."
    prompt_format = "System prompt. <|user|> {instruction} <|bot|> "
    expected_return = ["System prompt. <|user|> Write a short story. <|bot|> "]
    assert construct_prompts(text, prompt_format) == expected_return
    assert construct_prompts([text], prompt_format) == expected_return
    assert construct_prompts([Prompt(prompt=text)], prompt_format) == expected_return
