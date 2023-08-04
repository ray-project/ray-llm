from typer.testing import CliRunner

from aviary.cli import app

runner = CliRunner()


class TestCli:
    def test_metadata(self, aviary_testing_model):  # noqa: F811
        result = runner.invoke(
            app,
            [
                "metadata",
                "--model",
                aviary_testing_model,
            ],
        )
        assert result.exit_code == 0
        assert result.stdout

    def test_models(self, aviary_testing_model):  # noqa: F811
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert aviary_testing_model in result.stdout

    def test_query(self, aviary_testing_model):  # noqa: F811
        result = runner.invoke(
            app,
            [
                "query",
                "--print-stats",
                "--model",
                aviary_testing_model,
                "--prompt",
                "hello",
            ],
        )
        print(result.stdout)
        assert result.exit_code == 0
        assert result.stdout

    def test_multi_query(self, aviary_testing_model):  # noqa: F811
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
                aviary_testing_model,
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
