import itertools
import random
import time
import traceback
from functools import partial
from typing import Optional
from uuid import uuid4

import openai
import openai.error
import ray
import typer

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated
app = typer.Typer()


def _normal_completions_request(model: str, stream: bool):
    id = uuid4().hex
    print(f"Sending normal request to {model} ({id})")
    try:
        if stream:
            for completion in openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test",
                temperature=0.0,
                max_tokens=64,
                stream=True,
            ):
                print(completion, flush=True)
        else:
            completion = openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test",
                temperature=0.0,
                max_tokens=64,
            )
    except Exception as e:
        print(f"{model} {id} failed with exception:")
        traceback.print_exc()
        raise RuntimeError(f"{model} {id} failed with exception") from e
    print(completion, flush=True)
    assert completion.choices[
        0
    ].finish_reason, f"{model} {id} Should have a finish reason"
    if completion.choices[0].finish_reason == "length":
        assert completion.usage.completion_tokens == 64
    else:
        assert completion.usage.completion_tokens <= 64


def _bad_completions_request(model: str, stream: bool):
    # Send a bad request
    id = uuid4().hex
    print(f"Sending bad temperature request to {model} ({id})")
    try:
        if stream:
            for _c in openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test",
                temperature=-1.0,
                max_tokens=64,
                stream=True,
            ):
                pass
        else:
            openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test",
                temperature=-1.0,
                max_tokens=64,
            )
        raise RuntimeError(
            f"{model} {id} should have raised an exception for bad temperature."
        )
    except (openai.error.InvalidRequestError, openai.error.APIError) as e:
        if not stream:
            assert isinstance(e, openai.error.InvalidRequestError), (
                f"Exception {e} for too long prompt (non stream) should have "
                "been an InvalidRequestError."
            )
        assert "temperature" in str(
            e
        ), f"Exception {e} for bad temperature should have mentioned temperature."
        print(f"Exception {e} for bad temperature caught as expected.")


def _too_long_completion_request(model: str, stream: bool):
    # Send a too long prompt
    id = uuid4().hex
    print(f"Sending long prompt request to {model}")
    try:
        if stream:
            for _ in openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test" + " test " * 20000,
                temperature=0.0,
                max_tokens=64,
                stream=True,
            ):
                pass
        else:
            openai.Completion.create(
                model=model,
                prompt=f"{id} This is a test" + " test " * 20000,
                temperature=0.0,
                max_tokens=64,
            )
        raise RuntimeError(
            f"{model} {id} should have raised an exception for too long prompt."
        )
    except (openai.error.InvalidRequestError, openai.error.APIError) as e:
        if not stream:
            assert isinstance(e, openai.error.InvalidRequestError), (
                f"Exception {e} for too long prompt (non stream) should have "
                "been an InvalidRequestError."
            )
        assert (
            "prompt" in str(e).lower()
        ), f"Exception {e} for too long prompt should have mentioned prompt."
        print(f"Exception {e} for too long prompt caught as expected.")


def _normal_chat_request(model: str, stream: bool):
    id = uuid4().hex
    print(f"Sending normal request to {model} ({id})")
    try:
        if stream:
            for chat_completion in openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'."},
                ],
                temperature=0.0,
                max_tokens=64,
                stream=True,
            ):
                print(chat_completion, flush=True)
        else:
            chat_completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'."},
                ],
                temperature=0.0,
                max_tokens=64,
            )
    except Exception as e:
        print(f"{model} {id} failed with exception:")
        traceback.print_exc()
        raise RuntimeError(f"{model} {id} failed with exception") from e
    print(chat_completion, flush=True)
    assert chat_completion.choices[
        0
    ].finish_reason, f"{model} {id} Should have a finish reason"
    if chat_completion.choices[0].finish_reason == "length":
        assert chat_completion.usage.completion_tokens == 64
    else:
        assert chat_completion.usage.completion_tokens <= 64


def _bad_chat_request(model: str, stream: bool):
    # Send a bad request
    id = uuid4().hex
    print(f"Sending bad temperature request to {model} ({id})")
    try:
        if stream:
            for _ in openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'."},
                ],
                temperature=-1.0,
                max_tokens=64,
                stream=True,
            ):
                pass
        else:
            openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'."},
                ],
                temperature=-1.0,
                max_tokens=64,
            )
        raise RuntimeError(
            f"{model} {id} should have raised an exception for bad temperature."
        )
    except (openai.error.InvalidRequestError, openai.error.APIError) as e:
        if not stream:
            assert isinstance(e, openai.error.InvalidRequestError), (
                f"Exception {e} for too long prompt (non stream) should have "
                "been an InvalidRequestError."
            )
        assert "temperature" in str(
            e
        ), f"Exception {e} for bad temperature should have mentioned temperature."
        print(f"Exception {e} for bad temperature caught as expected.")


def _too_long_chat_request(model: str, stream: bool):
    # Send a too long prompt
    id = uuid4().hex
    print(f"Sending long prompt request to {model}")
    try:
        if stream:
            for _ in openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'." * 20000},
                ],
                temperature=0.0,
                max_tokens=64,
                stream=True,
            ):
                pass
        else:
            openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{id} You are a helpful assistant."},
                    {"role": "user", "content": "Say 'test'." * 20000},
                ],
                temperature=0.0,
                max_tokens=64,
            )
        raise RuntimeError(
            f"{model} {id} should have raised an exception for too long prompt."
        )
    except (openai.error.InvalidRequestError, openai.error.APIError) as e:
        if not stream:
            assert isinstance(e, openai.error.InvalidRequestError), (
                f"Exception {e} for too long prompt (non stream) should have "
                "been an InvalidRequestError."
            )
        assert (
            "prompt" in str(e).lower()
        ), f"Exception {e} for too long prompt should have mentioned prompt."
        print(f"Exception {e} for too long prompt caught as expected.")


def test_model(model, stream):
    _normal_chat_request(model, stream)
    _bad_chat_request(model, stream)
    _too_long_chat_request(model, stream)
    _normal_completions_request(model, stream)
    _bad_completions_request(model, stream)
    _too_long_completion_request(model, stream)


@ray.remote(num_cpus=0.1)
def test_model_random_order(model, url, api_key, seed):
    random.seed(seed)
    openai.api_base = url
    openai.api_key = api_key
    ops = (
        [_normal_chat_request] * 4
        + [_normal_completions_request] * 4
        + [
            _bad_chat_request,
            _too_long_chat_request,
            _bad_completions_request,
            _too_long_completion_request,
        ]
    )
    ops = [
        partial(op, stream=stream)
        for op, stream in itertools.product(ops, (True, False))
    ]
    random.shuffle(ops)
    for op in ops:
        op(model)
        time.sleep(0.05)


@app.command(name="run", help="Run integration tests on Endpoints API.")
def run(
    api_key: Annotated[str, typer.Option(help="Your Endpoints API key.")],
    url: Annotated[
        str, typer.Option(help="The Endpoints API URL.")
    ] = "https://api.endpoints.anyscale.com/v1",
    seed: Annotated[
        Optional[int], typer.Option(help="The random seed to use for the test.")
    ] = None,
):
    """Run integration tests on Endpoints API.

    Args:
        api_key: Your Endpoints API key.
        url: The Endpoints API URL.
        seed: The random seed to use for the test. If None, current
            timestamp will be used.
    """
    if seed is None:
        seed = int(time.monotonic())
    print(f"Starting test with seed {seed}...")
    random.seed(seed)
    openai.api_base = url
    openai.api_key = api_key
    # Get models
    models = openai.Model.list()
    models = [model.id for model in models.data]
    for model in models:
        test_model(model, stream=False)
        test_model(model, stream=True)

    # Parallel test
    print("Starting parallel test...")
    parallel_models = models * 4
    random.shuffle(parallel_models)
    ray.get(
        [
            test_model_random_order.remote(
                model, url=url, api_key=api_key, seed=random.randint(1, 100000)
            )
            for model in parallel_models
        ]
    )
    time.sleep(1)

    for model in models:
        test_model(model, stream=True)
        test_model(model, stream=False)

    print("", flush=True)
    print("Test finished successfully!", flush=True)


if __name__ == "__main__":
    app()
