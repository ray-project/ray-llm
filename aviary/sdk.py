import os
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

import openai

from aviary.common.models import ChatCompletion, Completion, Model
from aviary.common.utils import (
    _get_langchain_model,
    _is_aviary_model,
    assert_has_backend,
)

__all__ = [
    "Model",
    "Completion",
    "ChatCompletion",
    "models",
    "metadata",
    "completions",
    "run",
    "get_aviary_backend",
    "stream",
]


class AviaryResource:
    """Stores information about the Aviary backend configuration."""

    def __init__(self, backend_url: str, token: str):
        assert "::param" not in backend_url, "backend_url not set correctly"
        assert "::param" not in token, "token not set correctly"

        self.backend_url = backend_url
        self.token = token
        self.bearer = f"Bearer {token}" if token else ""


class URLNotSetException(Exception):
    pass


def get_aviary_backend(verbose: Optional[bool] = None):
    """
    Establishes a connection to the Aviary backed after establishing
    the information using environmental variables.

    For direct connection to the aviary backend (e.g. running on the same cluster),
    no AVIARY_TOKEN is required. Otherwise, the AVIARY_URL and AVIARY_TOKEN environment
    variables are required.

    Args:
        verbose: Whether to print the connecting message.

    Returns:
        An instance of the AviaryResource class.
    """
    aviary_url = os.getenv("AVIARY_URL", os.getenv("OPENAI_API_BASE"))
    if not aviary_url:
        raise URLNotSetException("AVIARY_URL or OPENAI_API_BASE must be set")

    aviary_token = os.getenv("AVIARY_TOKEN", os.getenv("OPENAI_API_KEY")) or ""

    aviary_url += "/v1" if not aviary_url.endswith("/v1") else ""

    if verbose is None:
        verbose = os.environ.get("AVIARY_SILENT", "0") == "0"
    if verbose:
        print(f"Connecting to Aviary backend at: {aviary_url}")
    return AviaryResource(aviary_url, aviary_token)


@contextmanager
def openai_aviary_context():
    backend = get_aviary_backend()
    original_api_base = openai.api_base
    original_api_key = openai.api_key
    openai.api_base = backend.backend_url
    openai.api_key = backend.token
    yield
    openai.api_base = original_api_base
    openai.api_key = original_api_key


@openai_aviary_context()
def models() -> List[str]:
    """List available models"""
    models = openai.Model.list()
    return [model.id for model in models.data]


@openai_aviary_context()
def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    metadata = openai.Model.retrieve(model_id)
    return metadata


def completions(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    **kwargs,
) -> Dict[str, Union[str, float, int]]:
    """Get completions from Aviary models."""
    kwargs.setdefault("max_tokens", None)
    if _is_aviary_model(model):
        with openai_aviary_context():
            if use_prompt_format:
                result = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    **kwargs,
                )
            else:
                result = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    stream=False,
                    **kwargs,
                )
            return result
    llm = _get_langchain_model(model)
    return llm.predict(prompt)


def query(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    **kwargs,
) -> Dict[str, Union[str, float, int]]:
    warnings.warn(
        "'query' is deprecated, please use 'completions' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return completions(model, prompt, use_prompt_format, **kwargs)


def stream(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    **kwargs,
) -> Iterator[Dict[str, Union[str, float, int]]]:
    """Query Aviary and stream response"""
    kwargs.setdefault("max_tokens", None)
    if _is_aviary_model(model):
        with openai_aviary_context():
            if use_prompt_format:
                result = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    **kwargs,
                )
            else:
                result = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    stream=True,
                    **kwargs,
                )
            return result
    else:
        # TODO implement streaming for langchain models
        raise RuntimeError("Streaming is currently only supported for aviary models")


def run(*model: str, blocking: bool = True) -> None:
    """Run Aviary on the local ray cluster

    args:
        *model: Models to run.
        blocking: Whether to block the CLI until the application is ready.

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from aviary.backend.server.run import run

    run(*model, blocking=blocking)


def shutdown() -> None:
    """Shutdown the Aviary backend server"""
    assert_has_backend()
    from ray import serve

    serve.shutdown()
