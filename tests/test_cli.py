import pytest
from typer.testing import CliRunner

from aviary.cli import app

runner = CliRunner()


TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"


def test_metadata():
    result = runner.invoke(
        app,
        [
            "metadata",
            "--model",
            TEST_MODEL,
        ],
    )
    assert result.exit_code == 0
    assert result.stdout


def test_models():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert TEST_MODEL in result.stdout


def test_query():
    result = runner.invoke(
        app,
        [
            "query",
            "--print-stats",
            "--model",
            TEST_MODEL,
            "--prompt",
            "hello",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert result.stdout


def test_batch_query():
    result = runner.invoke(
        app,
        [
            "query",
            "--print-stats",
            "--model",
            TEST_MODEL,
            "--prompt",
            "hello",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert result.stdout

    result = runner.invoke(
        app,
        [
            "query",
            "--model",
            TEST_MODEL,
            "--prompt",
            "hello",
            "--prompt",
            "world",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert result.stdout


def test_multi_query():
    prompts = [
        "What is the meaning of death?",
        "What is the best indie band of the 90s?",
    ]
    separator = "----"
    prompt_file = "prompts.txt"
    output_file = "aviary-output.json"

    with open(prompt_file, "w") as f:
        f.write(separator.join(prompts))

    result = runner.invoke(
        app,
        [
            "query",
            "--model",
            TEST_MODEL,
            "--prompt-file",
            prompt_file,
            "--separator",
            separator,
            "--output-file",
            output_file,
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0


# FIXME (max) add GPT4 key to GitHub secrets
@pytest.mark.skip(reason="GPT-4 not yet available in CI")
def test_eval():
    input_file = "aviary-output.json"
    eval_output = "evaluation-output.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input-file",
            input_file,
            "--evaluation-file",
            eval_output,
            "--evaluator",
            "gpt-4",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.skip(reason="Streaming is not yet available in CI")
def test_stream():
    result = runner.invoke(
        app,
        [
            "stream",
            "--model",
            TEST_MODEL,
            "--prompt",
            "hello",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert result.stdout
