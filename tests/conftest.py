import importlib
import os
import pathlib
import time
from unittest.mock import patch

import pytest
from ray import serve
from typer.testing import CliRunner

import rayllm.sdk
from rayllm.testing.mock_vllm_engine import MockVLLMEngine

runner = CliRunner()

try:
    from .endpoints.conftest import *  # noqa
except ImportError:
    pass


def get_test_model_path():
    current_file_dir = pathlib.Path(__file__).absolute().parent
    test_model_path = os.environ.get(
        "AVIARY_TEST_MODEL_PATH", current_file_dir / "mock_model.yaml"
    )
    test_model_path = pathlib.Path(test_model_path)

    if not test_model_path.exists():
        raise FileNotFoundError(f"Could not find {test_model_path}")
    return test_model_path


@pytest.fixture(scope="class")
def aviary_testing_model():
    test_model_path = get_test_model_path()
    test_model_runner = os.environ.get(
        "AVIARY_TEST_MODEL_LAUNCH_MODULE_PATH", "rayllm.backend.server.run"
    ).lower()
    test_model_patch_target = os.environ.get(
        "AVIARY_TEST_VLLM_PATCH_TARGET", test_model_runner
    )

    launch_fn = "run"
    runner_fn = getattr(importlib.import_module(test_model_runner), launch_fn)
    serve.shutdown()
    with patch.multiple(
        target=test_model_patch_target,
        VLLMEngine=MockVLLMEngine,
    ):
        aviary_url = runner_fn(vllm_base_args=[str(test_model_path.absolute())])

    openai_api_base = f"{aviary_url}/v1"
    openai_api_key = "not_an_actual_key"
    with patch.dict(
        os.environ,
        {
            "AVIARY_URL": aviary_url,
            "OPENAI_API_BASE": openai_api_base,
            "OPENAI_API_KEY": openai_api_key,
        },
    ):
        # Block until the deployment is ready
        # Wait at most 200s [3 min]
        for _i in range(20):
            try:
                model = rayllm.sdk.models()[0]
                assert model
                break
            except Exception as e:
                print("Error", e)
                pass
            time.sleep(10)
        yield model
    serve.shutdown()
