from typing import Any, Dict, List, Union

try:
    from langchain.llms import OpenAIChat

    LANGCHAIN_INSTALLED = True
    LANGCHAIN_SUPPORTED_PROVIDERS = {"openai": OpenAIChat}
except ImportError:
    LANGCHAIN_INSTALLED = False

from aviary.api.env import assert_has_backend

__all__ = ["models", "metadata", "query", "batch_query", "run"]


def models() -> List[str]:
    """List available models"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.models()


def _is_aviary_model(model: str) -> bool:
    """
    Determine if this is an aviary model. Aviary
    models do not have a '://' in them.
    """
    return "://" not in model


def _supports_batching(model: str) -> bool:
    provider, _ = model.split("://", 1)
    return provider != "openai"


def _get_langchain_model(model: str):
    if not LANGCHAIN_INSTALLED:
        raise ValueError(
            f"Unsupported model {model}. If you want to use a langchain-"
            "compatible model, install langchain ( pip install langchain )."
        )

    provider, model_name = model.split("://", 1)
    if provider not in LANGCHAIN_SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown model provider for {model}. Supported providers are: "
            f"{' '.join(LANGCHAIN_SUPPORTED_PROVIDERS.keys())}"
        )
    return LANGCHAIN_SUPPORTED_PROVIDERS[provider](model_name=model_name)


def _convert_to_aviary_format(model: str, llm_result):
    generation = llm_result.generations
    result_list = [{"generated_text": x.text} for x in generation[0]]
    return result_list


def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    from aviary.common.backend import get_aviary_backend

    backend = get_aviary_backend()
    return backend.metadata(model_id)


def query(model: str, prompt: str) -> Dict[str, Union[str, float, int]]:
    """Query Aviary"""
    from aviary.common.backend import get_aviary_backend

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        return backend.completions(prompt, model)
    llm = _get_langchain_model(model)
    return llm.predict(prompt)


def batch_query(
    model: str, prompts: List[str]
) -> List[Dict[str, Union[str, float, int]]]:
    """Batch Query Aviary"""
    from aviary.common.backend import get_aviary_backend

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        return backend.batch_completions(prompts, model)
    else:
        llm = _get_langchain_model(model)
        if _supports_batching(model):
            result = llm.generate(prompts)
            converted = _convert_to_aviary_format(model, result)
        else:
            result = [{"generated_text": llm.predict(prompt)} for prompt in prompts]
            converted = result
        return converted


def run(*model: str) -> None:
    """Run Aviary on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from aviary.backend.server.run import run

    run(*model)
