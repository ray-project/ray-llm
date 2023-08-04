from unittest.mock import MagicMock

from transformers import AutoTokenizer

from aviary.backend.llm.continuous.types import InferenceTask
from aviary.backend.llm.engine.tgi import (
    QuotaBasedTaskSelectionPolicy,
    TextGenerationInferenceEngine,
    TransformersTokenizer,
)
from aviary.backend.server.models import (
    GenerationConfig,
    PromptFormat,
    QuotaBasedTaskSelectionPolicyConfig,
    SchedulerConfig,
    TextGenerationInferenceEngineConfig,
)

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


def test_process_request():
    """Tests that the request is processed correctly and has the correct
    max_new_tokens value."""
    max_input_length = 1024
    max_total_tokens = 2048
    engine = TextGenerationInferenceEngine(
        TextGenerationInferenceEngineConfig(
            type="TextGenerationInferenceEngine",
            model_id=MODEL_ID,
            generation=GenerationConfig(
                prompt_format=PromptFormat(
                    system="{instruction}",
                    assistant="{instruction}",
                    trailing_assistant="",
                    user="{instruction}",
                )
            ),
            scheduler=SchedulerConfig(
                policy=QuotaBasedTaskSelectionPolicyConfig(
                    max_input_length=max_input_length, max_total_tokens=max_total_tokens
                )
            ),
        )
    )
    engine.tokenizer = TransformersTokenizer(AutoTokenizer.from_pretrained(MODEL_ID))
    engine.task_selection_policy = QuotaBasedTaskSelectionPolicy(
        **engine.engine_config.scheduler.policy.dict(exclude={"type"})
    )
    engine.inference_task_cls = InferenceTask
    engine.scheduler = MagicMock()

    # 1 input token
    task = engine.process_request(" ", max_new_tokens=None, sampling_params={})
    assert task.request.max_new_tokens == max_total_tokens - task.request.input_tokens

    task = engine.process_request(" ", max_new_tokens=2, sampling_params={})
    assert task.request.max_new_tokens == 2

    task = engine.process_request(
        " ", max_new_tokens=max_total_tokens + 1, sampling_params={}
    )
    assert task.request.max_new_tokens == max_total_tokens - task.request.input_tokens
