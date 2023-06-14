import os
from typing import Any, Dict, List, Union

import requests

from aviary.common.constants import TIMEOUT, DEFAULT_API_VERSION
from aviary.api.utils import (
    AviaryBackend,
    BackendError,
    assert_has_backend,
    _is_aviary_model,
    _supports_batching,
    _get_langchain_model,
    _convert_to_aviary_format,
)

__all__ = ["models", "metadata", "completions", "batch_completions", "run",
           "get_aviary_backend"]


def get_aviary_backend():
    """
    Establishes a connection to the Aviary backed after establishing
    the information using environmental variables.
    If the AVIARY_MOCK environmental variable is set, then a mock backend is used.

    For direct connection to the aviary backend (e.g. running on the same cluster),
    no AVIARY_TOKEN is required. Otherwise, the AVIARY_URL and AVIARY_TOKEN environment
    variables are required.

    Returns:
        backend: An instance of the Backend class.
    """
    aviary_url = os.getenv("AVIARY_URL")
    assert aviary_url, "AVIARY_URL must be set"

    aviary_token = os.getenv("AVIARY_TOKEN")
    assert aviary_token, "AVIARY_TOKEN must be set"

    bearer = f"Bearer {aviary_token}" if aviary_token else ""
    aviary_url += "/" if not aviary_url.endswith("/") else ""

    print("Connecting to Aviary backend at: ", aviary_url)
    return AviaryBackend(aviary_url, bearer)


def models(version: str = DEFAULT_API_VERSION) -> List[str]:
    """List available models"""
    backend = get_aviary_backend()
    request_url = backend.backend_url + version + "/models"
    response = requests.get(request_url, headers=backend.header, timeout=TIMEOUT)
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise BackendError(
            f"Error decoding JSON from {request_url}. Text response: {response.text}",
            response=response,
        ) from e
    return result


def metadata(model_id: str,
             version: str = DEFAULT_API_VERSION) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    backend = get_aviary_backend()
    url = backend.backend_url + version + "/metadata/" + model_id.replace("/", "--")
    response = requests.get(url, headers=backend.header, timeout=TIMEOUT)
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise BackendError(
            f"Error decoding JSON from {url}. Text response: {response.text}",
            response=response,
        ) from e
    return result


def completions(
        model: str,
        prompt: str,
        version: str = DEFAULT_API_VERSION
) -> Dict[str, Union[str, float, int]]:
    """Query Aviary"""

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + version + "/query/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json={"prompt": prompt},
            timeout=TIMEOUT,
        )
        try:
            return response.json()[model]
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
    llm = _get_langchain_model(model)
    return llm.predict(prompt)


def batch_completions(
        model: str,
        prompts: List[str],
        version: str = DEFAULT_API_VERSION
) -> List[Dict[str, Union[str, float, int]]]:
    """Batch Query Aviary"""

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + version + "/query/batch/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json=[{"prompt": prompt} for prompt in prompts],
            timeout=TIMEOUT,
        )
        try:
            return response.json()[model]
        except requests.JSONDecodeError as e:
            raise BackendError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
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
