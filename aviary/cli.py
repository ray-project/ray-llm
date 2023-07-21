import ast
import json

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

__all__ = ["app", "models", "metadata", "query", "batch_query", "run"]

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


def _print_result(result, model, print_stats):
    rp(f"[bold]{model}:[/]")
    if print_stats:
        rp("[bold]Stats:[/]")
        rp(result)
    else:
        rp(result["generated_text"])


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
            query_results = sdk.batch_completions(m, prompt)
            for result in query_results:
                _print_result(result, m, print_stats)

            for i, p in enumerate(prompt):
                result = query_results[i]
                text = result["generated_text"]
                del result["generated_text"]
                results[p].append({"model": m, "result": text, "stats": result})

        progress.add_task(description="Writing output file.", total=None)
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=2))


@app.command(deprecated=True, name="batch_query")
def batch_query(
    model: Annotated[List[str], model_type],
    prompt: Annotated[List[str], prompt_type],
    print_stats: Annotated[bool, stats_type] = False,
):
    """Query a model with a batch of prompts."""
    # TODO (max): deprecate and rename to "batch_completions" to match the API
    with progress_spinner() as progress:
        for m in model:
            progress.add_task(
                description=f"Processing prompt against {m}...", total=None
            )
            results = sdk.batch_completions(m, prompt)
            for result in results:
                _print_result(result, m, print_stats)


@app.command(deprecated=True, name="multi_query")
def multi_query(
    model: Annotated[List[str], model_type],
    prompt_file: Annotated[str, prompt_file_type],
    separator: Annotated[str, separator_type] = "----",
    output_file: Annotated[str, results_type] = "aviary-output.json",
):
    """Query one or multiple models with a batch of prompts taken from a file."""

    with progress_spinner() as progress:
        progress.add_task(
            description=f"Loading your prompts from {prompt_file}.", total=None
        )
        with open(prompt_file, "r") as f:
            prompts = f.read().split(separator)
        results = {prompt: [] for prompt in prompts}

        for m in model:
            progress.add_task(
                description=f"Processing all prompts against model: {model}.",
                total=None,
            )
            query_results = sdk.batch_completions(m, prompts)
            for i, prompt in enumerate(prompts):
                result = query_results[i]
                text = result["generated_text"]
                del result["generated_text"]
                results[prompt].append({"model": m, "result": text, "stats": result})

        progress.add_task(description="Writing output file.", total=None)
        with open(output_file, "w") as f:
            f.write(json.dumps(results, indent=2))


evaluator_type = typer.Option(help="Which LLM to use for evaluation")


@app.command()
def run(model: Annotated[List[str], model_type]):
    """Start a model in Aviary.

    Args:
        *model: The model to run.
    """
    sdk.run(*model)


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
    # TODO make this use the Response object
    num_input_tokens = 0
    num_input_tokens_batch = 0
    num_generated_tokens = 0
    num_generated_tokens_batch = 0
    preprocessing_time = 0
    generation_time = 0
    for chunk in sdk.stream(model, prompt):
        text = chunk["generated_text"]
        num_input_tokens = chunk["num_input_tokens"]
        num_input_tokens_batch = chunk["num_input_tokens_batch"]
        num_generated_tokens += chunk["num_generated_tokens"]
        num_generated_tokens_batch += chunk["num_generated_tokens_batch"]
        preprocessing_time += chunk["preprocessing_time"]
        generation_time += chunk["generation_time"]
        rp(text, end="")
    rp("")
    if print_stats:
        rp("[bold]Stats:[/]")
        chunk.pop("generated_text")
        total_time = preprocessing_time + generation_time
        num_total_tokens = num_generated_tokens + num_input_tokens
        num_total_tokens_batch = num_generated_tokens_batch + num_input_tokens_batch
        rp(
            {
                "num_input_tokens": num_input_tokens,
                "num_input_tokens_batch": num_input_tokens_batch,
                "num_generated_tokens": num_generated_tokens,
                "num_generated_tokens_batch": num_generated_tokens_batch,
                "preprocessing_time": preprocessing_time,
                "generation_time": generation_time,
                "generation_time_per_token": generation_time / num_total_tokens,
                "generation_time_per_token_batch": generation_time
                / num_total_tokens_batch,
                "num_total_tokens": num_total_tokens,
                "num_total_tokens_batch": num_total_tokens_batch,
                "total_time": total_time,
                "total_time_per_token": total_time / num_total_tokens,
                "total_time_per_token_batch": total_time / num_total_tokens_batch,
            }
        )


if __name__ == "__main__":
    app()
