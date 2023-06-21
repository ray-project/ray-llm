import logging
import os
from typing import Any, Dict, List, Optional, Union
import requests

from aviary.common.utils import (
    ResponseError,
    _convert_to_aviary_format,
    _get_langchain_model,
    _is_aviary_model,
    _supports_batching,
    assert_has_backend,
)
from aviary.common.constants import TIMEOUT
from aviary.common.models import Model, Completion, ChatCompletion


__all__ = [
    "Model",
    "Completion",
    "ChatCompletion",
    "models",
    "metadata",
    "completions",
    "batch_completions",
    "run",
    "get_aviary_backend",
]


class AviaryResource:
    """Stores information about the Aviary backend configuration.
    """
    def __init__(self, backend_url: str, bearer: str):
        assert "::param" not in backend_url, "backend_url not set correctly"
        assert "::param" not in bearer, "bearer not set correctly"

        self.backend_url = backend_url
        self.bearer = bearer
        self.header = {"Authorization": self.bearer}


def get_aviary_backend():
    """
    Establishes a connection to the Aviary backed after establishing
    the information using environmental variables.
    If the AVIARY_MOCK environmental variable is set, then a mock backend is used.

    For direct connection to the aviary backend (e.g. running on the same cluster),
    no AVIARY_TOKEN is required. Otherwise, the AVIARY_URL and AVIARY_TOKEN environment
    variables are required.

    Returns:
        An instance of the AviaryResource class.
    """
    aviary_url = os.getenv("AVIARY_URL")
    assert aviary_url, "AVIARY_URL must be set"

    aviary_token = os.getenv("AVIARY_TOKEN")
    assert aviary_token, "AVIARY_TOKEN must be set"

    bearer = f"Bearer {aviary_token}" if aviary_token else ""
    aviary_url += "/" if not aviary_url.endswith("/") else ""

    print("Connecting to Aviary backend at: ", aviary_url)
    return AviaryResource(aviary_url, bearer)


def _get_result(response: requests.Response) -> Dict[str, Any]:
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise ResponseError(
            f"Error decoding JSON from {response.url}. Text response: {response.text}",
            response=response,
        ) from e
    return result


def model_list(cls) -> Model:
    """List all available Aviary models"""
    backend = get_aviary_backend()
    request_url = backend.backend_url + "v1/models"
    response = requests.get(request_url, headers=backend.header, timeout=TIMEOUT)
    result = _get_result(response)
    return Model(**result)


Model.list = classmethod(model_list)


def completion_create(
        cls,
        model: str,
        prompt: str,
        use_prompt_format: bool = True,
) -> Completion:
    """Create a completion from a prompt.
    """
    backend = get_aviary_backend()
    url = backend.backend_url + "v1/completions/" + model.replace("/", "--")
    response = requests.post(
        url,
        headers=backend.header,
        json={"prompt": prompt, "use_prompt_format": use_prompt_format},
        timeout=TIMEOUT,
    )
    return Completion(**_get_result(response))


Completion.create = classmethod(completion_create)


def chat_completion_create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        use_prompt_format: bool = True,
) -> ChatCompletion:
    """Create a chat completion from a list of messages.
    """
    backend = get_aviary_backend()
    url = backend.backend_url + "v1/chat/completions/" + model.replace("/", "--")
    response = requests.post(
        url,
        headers=backend.header,
        json={"messages": messages, "use_prompt_format": use_prompt_format},
        timeout=TIMEOUT,
    )
    return ChatCompletion(**_get_result(response))


ChatCompletion.create = classmethod(chat_completion_create)


def models() -> List[str]:
    """List available models"""
    backend = get_aviary_backend()
    request_url = backend.backend_url + "models"
    response = requests.get(request_url, headers=backend.header, timeout=TIMEOUT)
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise ResponseError(
            f"Error decoding JSON from {response.url}. Text response: {response.text}",
            response=response,
        ) from e
    return result


def metadata(
    model_id: str
) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    backend = get_aviary_backend()
    url = backend.backend_url + "metadata/" + model_id.replace("/", "--")
    response = requests.get(url, headers=backend.header, timeout=TIMEOUT)
    try:
        result = response.json()
    except requests.JSONDecodeError as e:
        raise ResponseError(
            f"Error decoding JSON from {url}. Text response: {response.text}",
            response=response,
        ) from e
    return result


def completions(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
) -> Dict[str, Union[str, float, int]]:
    """Get completions from Aviary models."""

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + "query/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json={"prompt": prompt, "use_prompt_format": use_prompt_format},
            timeout=TIMEOUT,
        )
        try:
            print(response.json())
            return response.json()[model]
        except requests.JSONDecodeError as e:
            raise ResponseError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
    llm = _get_langchain_model(model)
    return llm.predict(prompt)


def query(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
) -> Dict[str, Union[str, float, int]]:
    logging.warning("'query' is deprecated, please use 'completions' instead")
    return completions(model, prompt, use_prompt_format)


def batch_completions(
    model: str,
    prompts: List[str],
    use_prompt_format: Optional[List[bool]] = None,
) -> List[Dict[str, Union[str, float, int]]]:
    """Get batch completions from Aviary models."""

    if not use_prompt_format:
        use_prompt_format = [True] * len(prompts)

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + "query/batch/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json=[
                {"prompt": prompt, "use_prompt_format": use_format}
                for prompt, use_format in zip(prompts, use_prompt_format)
            ],
            timeout=TIMEOUT,
        )
        try:
            return response.json()[model]
        except requests.JSONDecodeError as e:
            raise ResponseError(
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


def batch_query(
    model: str,
    prompts: List[str],
    use_prompt_format: Optional[List[bool]] = None,
) -> List[Dict[str, Union[str, float, int]]]:
    logging.warning(
        "'batch_query' is deprecated, please use " "'batch_completions' instead"
    )
    return batch_completions(model, prompts, use_prompt_format)


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


Model.deploy = classmethod(run)
