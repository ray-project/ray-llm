# Mocks the API for testing purposes
from typing import Any, Dict, List, Union


def models() -> List[str]:
    """List available models"""
    return ["A", "B", "C"]


def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    return {
        "metadata": {
            "model_config": {
                "model_id": model_id,
                "model_url": f"https://huggingface.co/org/{model_id}",
                "model_description": f"This is a model description for model {model_id}"
            }
        }
    }


def completions(model: str, prompt: str) -> Dict[str, Union[str, float, int]]:
    """Query Aviary"""
    return {
        "generated_text": prompt,
        "total_time": 99,
        "num_total_tokens": 42.3,
    }


def batch_completions(
    model: str, prompts: List[str]
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


def run(*model: str) -> None:
    pass
