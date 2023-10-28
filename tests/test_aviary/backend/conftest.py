import pytest

from rayllm.backend.llm.vllm.vllm_engine import VLLMEngine
from rayllm.backend.llm.vllm.vllm_models import VLLMApp, VLLMEngineConfig
from rayllm.backend.server.models import GenerationConfig, PromptFormat, ScalingConfig

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"


class EngineGenerator:
    def __init__(self, max_input_length: int = 1024, max_total_tokens: int = 2048):
        self.max_input_length = max_input_length
        self.max_total_tokens = max_total_tokens

    def create_engine_config(self):
        return VLLMEngineConfig(
            type="VLLMEngine",
            model_id=MODEL_ID,
            generation=GenerationConfig(
                prompt_format=PromptFormat(
                    system="{instruction}",
                    assistant="{instruction}",
                    trailing_assistant="",
                    user="{instruction}",
                )
            ),
        )

    def create_scaling_config(self):
        return ScalingConfig(
            num_workers=1,
            num_gpus_per_worker=0,
            num_cpus_per_worker=1,
            placement_strategy="STRICT_PACK",
        )

    def create_engine(self):
        engine = VLLMEngine(
            VLLMApp(
                engine_config=self.create_engine_config(),
                scaling_config=self.create_scaling_config(),
            )
        )
        return engine


@pytest.fixture
def engine():
    return EngineGenerator().create_engine()


@pytest.fixture
def engine_generator():
    return EngineGenerator()
