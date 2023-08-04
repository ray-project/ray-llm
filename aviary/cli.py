import ast
import json

import requests

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from typing import List, Optional

import typer
from rich import print as rp
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from aviary import sdk
from aviary.common.evaluation import GPT

__all__ = ["app", "models", "metadata", "query", "run"]

app = typer.Typer()

model_type = typer.Option(
    default=..., help="The model to use. You can specify multiple models."
)
prompt_type = typer.Option(help="Prompt to query")
stats_type = typer.Option(help="Whether to print generated statistics")
prompt_file_type = typer.Option(
    default=..., help="File containing prompts. A simple text file"
)
separator_type = typer.Option(help="Separator used in prompt files")
results_type = typer.Option(help="Where to save the results")
true_or_false_type = typer.Option(default=False, is_flag=True)


@app.command()
def models(metadata: Annotated[bool, "Whether to print metadata"] = False):
    """Get a list of the available models"""
    result = sdk.models()
    if metadata:
        for model in result:
            rp(f"[bold]{model}:[/]")
            rp(sdk.metadata(model))
    else:
        print("\n".join(result))


@app.command()
def metadata(model: Annotated[List[str], model_type]):
    """Get metadata for models."""
    results = [sdk.metadata(m) for m in model]
    rp(results)


def _get_text(result: dict) -> str:
    if "text" in result["choices"][0]:
        return result["choices"][0]["text"]
    elif "message" in result["choices"][0]:
        return result["choices"][0]["message"]["content"]
    elif "delta" in result["choices"][0]:
        return result["choices"][0]["delta"].get("content", "")


def _print_result(result, model, print_stats):
    rp(f"[bold]{model}:[/]")
    if print_stats:
        rp("[bold]Stats:[/]")
        rp(result)
    else:
        rp(_get_text(result))


def progress_spinner():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    )


@app.command()
def query(
    model: Annotated[List[str], model_type],
    prompt: Annotated[Optional[List[str]], prompt_type] = None,
    prompt_file: Annotated[Optional[str], prompt_file_type] = None,
    separator: Annotated[str, separator_type] = "----",
    output_file: Annotated[str, results_type] = "aviary-output.json",
    print_stats: Annotated[bool, stats_type] = False,
):
    """Query one or several models with one or multiple prompts,
    optionally read from file, and save the results to a file."""
    # TODO (max): deprecate and rename to "completions" to match the API
    with progress_spinner() as progress:
        if prompt_file:
            with open(prompt_file, "r") as f:
                prompt = f.read().split(separator)

        results = {p: [] for p in prompt}

        for m in model:
            progress.add_task(
                description=f"Processing all prompts against model: {m}.",
                total=None,
            )
            query_results = [sdk.query(m, p) for p in prompt]
            for result in query_results:
                _print_result(result, m, print_stats)

            for i, p in enumerate(prompt):
                result = query_results[i]
                text = _get_text(result)
                results[p].append({"model": m, "result": text, "stats": result})

        progress.add_task(description="Writing output file.", total=None)
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=2))


def _get_yes_or_no_input(prompt) -> bool:
    while True:
        user_input = input(prompt).strip().lower()
        if user_input == "yes" or user_input == "y":
            return True
        elif user_input == "no" or user_input == "n" or user_input == "":
            return False
        else:
            print("Invalid input. Please enter 'yes / y' or 'no / n'.")


@app.command()
def run(
    model: Annotated[List[str], model_type],
    blocking: bool = True,
    restart: bool = true_or_false_type,
):
    """Start a model in Aviary.

    Args:
        *model: Models to run.
        blocking: Whether to block the CLI until the application is ready.
        restart: Whether to restart Aviary if it is already running.
    """
    msg = (
        "Running `aviary run` while Aviary is running will stop any exisiting Aviary (or other Ray Serve) deployments "
        f"and run the specified ones ({model}).\n"
        "Do you want to continue? [y/N]\n"
    )
    try:
        backend = sdk.get_aviary_backend(verbose=False)
        aviary_url = backend.backend_url
        aviary_started = False
        if aviary_url:
            health_check_url = f"{aviary_url}/health_check"
            aviary_started = requests.get(health_check_url).status_code == 200
        if aviary_started:
            if restart:
                restart_aviary = True
            else:
                restart_aviary = _get_yes_or_no_input(msg) or False

            if not restart_aviary:
                return
    except (requests.exceptions.ConnectionError, sdk.URLNotSetException):
        pass  # Aviary is not running

    sdk.shutdown()
    sdk.run(*model, blocking=blocking)


@app.command()
def shutdown():
    """Shutdown Aviary."""
    sdk.shutdown()


evaluator_type = typer.Option(help="Which LLM to use for evaluation")


@app.command()
def evaluate(
    input_file: Annotated[str, results_type] = "aviary-output.json",
    evaluation_file: Annotated[str, results_type] = "evaluation-output.json",
    evaluator: Annotated[str, evaluator_type] = "gpt-4",
):
    """Evaluate and summarize the results of a multi_query run with a strong
    'evaluator' LLM like GPT-4.
    The results of the ranking are stored to file and displayed in a table.
    """
    with progress_spinner() as progress:
        progress.add_task(description="Loading the evaluator LLM.", total=None)
        if evaluator == "gpt-4":
            eval_model = GPT()
        else:
            raise NotImplementedError(f"No evaluator for {evaluator}")

        with open(input_file, "r") as f:
            results = json.load(f)

        for prompt, result_list in results.items():
            progress.add_task(
                description=f"Evaluating results for prompt: {prompt}.", total=None
            )
            evaluation = eval_model.evaluate_results(prompt, result_list)
            try:
                # GPT-4 returns a string with a Python dictionary, hopefully!
                evaluation = ast.literal_eval(evaluation)
            except Exception:
                print(f"Could not parse evaluation: {evaluation}")

            for i, _res in enumerate(results[prompt]):
                results[prompt][i]["rank"] = evaluation[i]["rank"]

        progress.add_task(description="Storing evaluations.", total=None)
        with open(evaluation_file, "w") as f:
            f.write(json.dumps(results, indent=2))

    for prompt in results.keys():
        table = Table(title="Evaluation results (higher ranks are better)")

        table.add_column("Model", justify="left", style="cyan", no_wrap=True)
        table.add_column("Rank", style="magenta")
        table.add_column("Response", justify="right", style="green")

        for i, _res in enumerate(results[prompt]):
            model = results[prompt][i]["model"]
            response = results[prompt][i]["result"]
            rank = results[prompt][i]["rank"]
            table.add_row(model, str(rank), response)

        console = Console()
        console.print(table)


@app.command()
def stream(
    model: Annotated[str, model_type],
    prompt: Annotated[str, prompt_type],
    print_stats: Annotated[bool, stats_type] = False,
):
    """"""
    for chunk in sdk.stream(model, prompt):
        text = _get_text(chunk)
        rp(text, end="")
    rp("")
    if print_stats:
        rp("[bold]Stats:[/]")
        rp(chunk)


if __name__ == "__main__":
    app()
