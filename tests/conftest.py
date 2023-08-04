import contextlib
import os
import pathlib
import subprocess
import time
from unittest.mock import patch

import pytest
from ray import serve
from typer.testing import CliRunner

import aviary.sdk
from aviary.cli import app

runner = CliRunner()

try:
    from .endpoints.conftest import *  # noqa
except ImportError:
    pass


@pytest.fixture(scope="class")
def aviary_testing_model():
    current_file_dir = pathlib.Path(__file__).absolute().parent
    test_model_path = os.environ.get(
        "AVIARY_TEST_MODEL_PATH",
        current_file_dir / "models" / "hf-internal-testing--tiny-random-gpt2-cb.yaml",
    )
    test_model_path = pathlib.Path(test_model_path)
    test_model_runner = os.environ.get("AVIARY_TEST_MODEL_RUNNER", "aviary_run").lower()
    assert test_model_runner in ["aviary_run", "serve_run", "ignore"]
    if test_model_runner == "ignore":
        yield None
    else:
        if not test_model_path.exists():
            raise FileNotFoundError(f"Could not find {test_model_path}")
        test_model_path = str(test_model_path.absolute())
        if test_model_runner == "aviary_run":
            runner.invoke(
                app,
                [
                    "run",
                    "--model",
                    test_model_path,
                ],
            )
            aviary_url = "http://localhost:8000"
        elif test_model_runner == "serve_run":
            subprocess.run(
                ["serve", "run", test_model_path, "--non-blocking"], check=True
            )
            aviary_url = "http://localhost:8000/m"
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
            while True:
                try:
                    with contextlib.redirect_stdout(None):
                        model = aviary.sdk.models()[0]
                    assert model
                    break
                except Exception:
                    pass
                time.sleep(1)
            yield model
        serve.shutdown()
