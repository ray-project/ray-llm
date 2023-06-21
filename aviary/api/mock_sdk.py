# Mocks the API for testing purposes
from typing import Any, Dict, Iterator, List, Optional, Union

from aviary.common.constants import DEFAULT_API_VERSION


def models(version: str = DEFAULT_API_VERSION) -> List[str]:
    """List available models"""
    return ["A", "B", "C"]


def metadata(
    model_id: str, version: str = DEFAULT_API_VERSION
) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    return {
        "metadata": {
            "model_config": {
                "model_id": model_id,
                "model_url": f"https://huggingface.co/org/{model_id}",
                "model_description": f"This is a model description for model {model_id}",
            }
        }
    }


def completions(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    version: str = DEFAULT_API_VERSION,
) -> Dict[str, Union[str, float, int]]:
    """Query Aviary"""
    return {
        "generated_text": prompt,
        "total_time": 99,
        "num_total_tokens": 42.3,
    }


def batch_completions(
    model: str,
    prompts: List[str],
    use_prompt_format: Optional[List[bool]] = None,
    version: str = DEFAULT_API_VERSION,
) -> List[Dict[str, Union[str, float, int]]]:
    """Batch Query Aviary"""
    return [
        {
            "generated_text": prompt,
            "total_time": 99,
            "num_total_tokens": 42.3,
        }
        for prompt in prompts
    ]


def stream(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    version: str = DEFAULT_API_VERSION,
) -> Iterator[Dict[str, Union[str, float, int]]]:
    """Query Aviary and stream response"""
    for chunk in prompt.split():
        yield {
            "generated_text": chunk + " ",
            "total_time": 99,
            "generation_time": 99,
            "num_total_tokens": 42,
            "num_generated_tokens": 42,
        }


def run(*model: str) -> None:
    pass
