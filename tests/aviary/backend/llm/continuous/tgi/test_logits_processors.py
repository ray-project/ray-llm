import torch

from aviary.backend.llm.continuous.tgi.logits_processors import (
    HeterogeneousFrequencyPresencePenaltyLogitsProcessor,
    MinNewTokensLogitsProcessor,
)


def test_min_new_tokens_logits_processor():
    # This is a stateful processor.
    processor = MinNewTokensLogitsProcessor(
        generated_tokens=0, min_new_tokens=2, eos_token_id=0
    )
    assert processor(torch.LongTensor([]), torch.tensor([[1, 0.1, 0.1]]))[
        0, 0
    ] == float("-inf")
    assert processor(torch.LongTensor([1]), torch.tensor([[1, 0.1, 0.1]]))[
        0, 0
    ] == float("-inf")
    assert (
        processor(torch.LongTensor([1, 1]), torch.tensor([[1, 0.1, 0.1]]))[0, 0] == 1.0
    )
    assert (
        processor(torch.LongTensor([1, 1, 1]), torch.tensor([[1, 0.1, 0.1]]))[0, 0]
        == 1.0
    )


def test_min_new_tokens_logits_processor_generated_tokens_greater_than_0():
    processor = MinNewTokensLogitsProcessor(
        generated_tokens=1, min_new_tokens=2, eos_token_id=0
    )
    assert processor(torch.LongTensor([1]), torch.tensor([[1, 0.1, 0.1]]))[
        0, 0
    ] == float("-inf")
    assert (
        processor(torch.LongTensor([1, 1]), torch.tensor([[1, 0.1, 0.1]]))[0, 0] == 1.0
    )


def test_min_new_tokens_logits_processor_different_eos_id():
    processor = MinNewTokensLogitsProcessor(
        generated_tokens=0, min_new_tokens=2, eos_token_id=1
    )
    assert processor(torch.LongTensor([1]), torch.tensor([[1, 0.1, 0.1]]))[
        0, 1
    ] == float("-inf")


def test_min_new_tokens_logits_processor_min_new_tokens_0():
    processor = MinNewTokensLogitsProcessor(
        generated_tokens=0, min_new_tokens=0, eos_token_id=0
    )
    assert processor(torch.LongTensor([1]), torch.tensor([[1, 0.1, 0.1]]))[0, 0] == 1.0


def test_heterogeneous_frequency_presence_penalty_logits_processor_noop():
    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.0, 0.0, 0.0],
        frequency_penalty=[0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.LongTensor([[1, 1, 1], [1, 0, 0], [1, 2, 3]])
    scores = torch.tensor(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    )
    assert processor(input_ids, scores).allclose(
        torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
    )


def test_heterogeneous_frequency_presence_penalty_logits_processor_presence_penalty():
    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.1, 0.2, 0.3],
        frequency_penalty=[0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.LongTensor([[1, 1, 1], [1, 0, 0], [1, 2, 3]])
    scores = torch.tensor(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    )
    assert processor(input_ids, scores).allclose(
        torch.tensor(
            [[0.1, 0, 0.1, 0.1], [-0.1, -0.1, 0.1, 0.1], [0.1, -0.2, -0.2, -0.2]]
        )
    )


def test_heterogeneous_frequency_presence_penalty_logits_processor_frequency_penalty():
    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.0, 0.0, 0.0],
        frequency_penalty=[0.1, 0.2, 0.3],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.LongTensor([[1, 1, 1], [1, 0, 0], [1, 2, 3]])
    scores = torch.tensor(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    )
    assert processor(input_ids, scores).allclose(
        torch.tensor(
            [[0.1, -0.2, 0.1, 0.1], [-0.3, -0.1, 0.1, 0.1], [0.1, -0.2, -0.2, -0.2]]
        )
    )


def test_heterogeneous_frequency_presence_penalty_logits_processor_both_penalties():
    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.1, 0.2, 0.3],
        frequency_penalty=[0.1, 0.2, 0.3],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    input_ids = torch.LongTensor([[1, 1, 1], [1, 0, 0], [1, 2, 3]])
    scores = torch.tensor(
        [[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]
    )
    assert processor(input_ids, scores).allclose(
        torch.tensor(
            [[0.1, -0.3, 0.1, 0.1], [-0.5, -0.3, 0.1, 0.1], [0.1, -0.5, -0.5, -0.5]]
        )
    )


def test_heterogeneous_frequency_presence_penalty_logits_processor_test_filter():
    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.1, 0.2, 0.3],
        frequency_penalty=[0.4, 0.5, 0.6],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    processor = processor.filter([0, 2])
    assert processor.presence_penalty_tensor.allclose(
        torch.tensor([0.1, 0.3]).unsqueeze(1)
    )
    assert processor.frequency_penalty_tensor.allclose(
        torch.tensor([0.4, 0.6]).unsqueeze(1)
    )

    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.0, 0.2, 0.0],
        frequency_penalty=[0.0, 0.5, 0.1],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    processor = processor.filter([0, 2])
    assert processor.presence_penalty_tensor.allclose(
        torch.tensor([0.0, 0.0]).unsqueeze(1)
    )
    assert processor.frequency_penalty_tensor.allclose(
        torch.tensor([0.0, 0.1]).unsqueeze(1)
    )

    processor = HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
        presence_penalty=[0.0, 0.2, 0.0],
        frequency_penalty=[0.0, 0.5, 0.0],
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    processor = processor.filter([0, 2])
    assert processor is None
