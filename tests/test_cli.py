from typer.testing import CliRunner

from aviary.api.cli import app

runner = CliRunner()


def test_list_models():
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "mosaicml/mpt-7b-instruct" in result.stdout


def test_query():
    result = runner.invoke(
        app,
        [
            "query",
            "--print-stats",
            "--model",
            "amazon/LightGPT",
            "--prompt",
            "hello",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout

    result = runner.invoke(
        app,
        [
            "query",
            "--model",
            "amazon/LightGPT",
            "--model",
            "RWKV/rwkv-raven-14b",
            "--prompt",
            "hello",
        ],
    )
    assert result.exit_code == 0
    assert "amazon/LightGPT" in result.stdout
    assert "RWKV/rwkv-raven-14b" in result.stdout


def test_batch_query():
    result = runner.invoke(
        app,
        [
            "query",
            "--print-stats",
            "--model",
            "amazon/LightGPT",
            "--prompt",
            "hello",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout

    result = runner.invoke(
        app,
        [
            "query",
            "--model",
            "amazon/LightGPT",
            "--prompt",
            "hello",
            "--prompt",
            "world",
        ],
    )
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
            "amazon/LightGPT",
            "--model",
            "mosaicml/mpt-7b-instruct",
            "--model",
            "RWKV/rwkv-raven-14b",
            "--prompt-file",
            prompt_file,
            "--separator",
            separator,
            "--output-file",
            output_file,
        ],
    )

    assert result.exit_code == 0


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
