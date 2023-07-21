import json
import os
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union

import pydantic
import requests

from aviary.common.constants import TIMEOUT
from aviary.common.models import ChatCompletion, Completion, Model
from aviary.common.utils import (
    ResponseError,
    _convert_to_aviary_format,
    _get_langchain_model,
    _is_aviary_model,
    _supports_batching,
    assert_has_backend,
)

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
    "stream",
]


class AviaryResource:
    """Stores information about the Aviary backend configuration."""

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

    aviary_token = os.getenv("AVIARY_TOKEN", "")

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
    if result.get("error"):
        raise ResponseError(result["error"], response=response)
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
    max_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    frequency_penalty: float = 0.0,
) -> Completion:
    """Create a completion from a prompt."""
    backend = get_aviary_backend()
    url = backend.backend_url + "v1/completions/" + model.replace("/", "--")
    if stream:

        def gen():
            response = requests.post(
                url,
                headers=backend.header,
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "stopping_sequences": stop,
                    "stream": True,
                },
                timeout=TIMEOUT,
                stream=True,
            )
            chunk = ""
            try:
                for chunk in response.iter_lines(chunk_size=None, decode_unicode=True):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    data = json.loads(chunk)
                    if data.get("error"):
                        raise ResponseError(data["error"], response=response)
                    try:
                        yield Completion(**data)
                    except pydantic.ValidationError as e:
                        raise ResponseError(
                            f"Error decoding response from {response.url}.",
                            response=response,
                        ) from e
            except ConnectionError as e:
                raise ResponseError(str(e) + "\n" + chunk, response=response) from e

        return gen()
    else:
        response = requests.post(
            url,
            headers=backend.header,
            json={
                "prompt": prompt,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "stopping_sequences": stop,
            },
            timeout=TIMEOUT,
        )
        try:
            return Completion(**_get_result(response))
        except pydantic.ValidationError as e:
            raise ResponseError(
                f"Error decoding response from {response.url}.",
                response=response,
            ) from e


Completion.create = classmethod(completion_create)


def chat_completion_create(
    cls,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    top_p: float = 1.0,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    frequency_penalty: float = 0.0,
) -> ChatCompletion:
    """Create a chat completion from a list of messages."""
    backend = get_aviary_backend()
    url = backend.backend_url + "v1/chat/completions/" + model.replace("/", "--")
    if stream:

        def gen():
            response = requests.post(
                url,
                headers=backend.header,
                json={
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "stopping_sequences": stop,
                    "stream": True,
                },
                timeout=TIMEOUT,
                stream=True,
            )
            chunk = ""
            try:
                for chunk in response.iter_lines(chunk_size=None, decode_unicode=True):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    data = json.loads(chunk)
                    if data.get("error"):
                        raise ResponseError(data["error"], response=response)
                    try:
                        yield ChatCompletion(**data)
                    except pydantic.ValidationError as e:
                        raise ResponseError(
                            f"Error decoding response from {response.url}.",
                            response=response,
                        ) from e
            except ConnectionError as e:
                raise ResponseError(str(e) + "\n" + chunk, response=response) from e

        return gen()
    else:
        response = requests.post(
            url,
            headers=backend.header,
            json={
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "stopping_sequences": stop,
            },
            timeout=TIMEOUT,
        )
        try:
            return ChatCompletion(**_get_result(response))
        except pydantic.ValidationError as e:
            raise ResponseError(
                f"Error decoding response from {response.url}.",
                response=response,
            ) from e


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


def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
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
    **kwargs,
) -> Dict[str, Union[str, float, int]]:
    """Get completions from Aviary models."""

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + "query/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json={"prompt": prompt, "use_prompt_format": use_prompt_format, **kwargs},
            timeout=TIMEOUT,
        )
        try:
            response = response.json()
        except requests.JSONDecodeError as e:
            raise ResponseError(
                f"Error decoding JSON from {url}. Text response: {response.text}",
                response=response,
            ) from e
        if response.get("error"):
            raise ResponseError(response["error"], response=response)
        return response
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


def batch_completions(
    model: str,
    prompts: List[str],
    use_prompt_format: Optional[List[bool]] = None,
    kwargs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Union[str, float, int]]]:
    """Get batch completions from Aviary models."""

    if not kwargs:
        kwargs = [{}] * len(prompts)

    if not use_prompt_format:
        use_prompt_format = [True] * len(prompts)

    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + "query/batch/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json=[
                {"prompt": prompt, "use_prompt_format": use_format, **kwarg}
                for prompt, use_format, kwarg in zip(prompts, use_prompt_format, kwargs)
            ],
            timeout=TIMEOUT,
        )
        try:
            return response.json()
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


def stream(
    model: str,
    prompt: str,
    use_prompt_format: bool = True,
    **kwargs,
) -> Iterator[Dict[str, Union[str, float, int]]]:
    """Query Aviary and stream response"""
    if _is_aviary_model(model):
        backend = get_aviary_backend()
        url = backend.backend_url + "stream/" + model.replace("/", "--")
        response = requests.post(
            url,
            headers=backend.header,
            json={"prompt": prompt, "use_prompt_format": use_prompt_format, **kwargs},
            timeout=TIMEOUT,
            stream=True,
        )
        chunk = ""
        try:
            for chunk in response.iter_lines(chunk_size=None, decode_unicode=True):
                chunk = chunk.strip()
                if not chunk:
                    continue
                data = json.loads(chunk)
                if data.get("error"):
                    raise ResponseError(data["error"], response=response)
                yield data
        except ConnectionError as e:
            raise ResponseError(str(e) + "\n" + chunk, response=response) from e
    else:
        # TODO implement streaming for langchain models
        raise RuntimeError("Streaming is currently only supported for aviary models")


def batch_query(
    model: str,
    prompts: List[str],
    use_prompt_format: Optional[List[bool]] = None,
    kwargs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Union[str, float, int]]]:
    warnings.warn(
        "'batch_query' is deprecated, please use " "'batch_completions' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return batch_completions(model, prompts, use_prompt_format, kwargs)


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
