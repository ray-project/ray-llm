from typing import Any, Dict, List, Union

from aviary.api.env import assert_has_backend

__all__ = ["models", "metadata", "query", "batch_query", "run"]


def models() -> List[str]:
    """List available models"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.models()


def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.metadata(model_id)


def query(model: str, prompt: str) -> Dict[str, Union[str, float, int]]:
    """Query Aviary"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.completions(prompt, model)


def batch_query(
    model: str, prompts: List[str]
) -> List[Dict[str, Union[str, float, int]]]:
    """Batch Query Aviary"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.batch_completions(prompts, model)


def run(*model: str) -> None:
    """Run Aviary on the local ray cluster"""
    assert_has_backend()
    from aviary.backend.server.run import run

    run(*model)
