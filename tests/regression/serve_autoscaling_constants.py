import argparse
import os
import pathlib
import subprocess
from collections import deque
from typing import Any, Dict, Optional

import yaml
from regression_test_runner import RegressionTestBase
from users import create_completions_request, create_user
from utils import EndpointResults, get_queue_size, upload_to_s3, wait_for_queue_drain

parser = argparse.ArgumentParser(
    prog="Serve Autoscaling Constants",
    description=(
        "determine the maximum number of concurrent users that "
        "can be supported by the model"
    ),
)

parser.add_argument("--model-yaml-path", type=str, help="The model yaml to use.")
parser.add_argument(
    "--results-dir",
    default="",
    metavar="r",
    type=str,
    help="The directory to save the results to.",
)
parser.add_argument(
    "--s3-bucket", default="", type=str, help="The s3 bucket to save the results to."
)


def get_max_concurrent_users(
    model: str,
    prompt: str,
    num_output_tokens: int,
    sampling_params: Optional[Dict[str, Any]] = None,
) -> int:
    request_cls = create_completions_request(
        model=model,
        prompts=[prompt],
        num_output_tokens=num_output_tokens,
        sampling_params=sampling_params,
    )
    user_cls = create_user(requests=request_cls)
    test_runner = RegressionTestBase(user_cls)

    test_runner.set_num_users(num_users=150, spawn_rate=0.5)

    queue_sizes = deque(maxlen=100)

    while True:
        model_queue_size = get_queue_size(model_id=model)
        queue_sizes.append(model_queue_size)

        average_queue_size = sum(queue_sizes) / len(queue_sizes)

        if average_queue_size > 10:
            break

    max_concurrent_users = test_runner.get_curr_num_users()
    test_runner.shutdown()
    return max_concurrent_users


if __name__ == "__main__":
    args = parser.parse_args()

    path = pathlib.Path(args.model_yaml_path)
    assert path.exists()

    model_yaml = yaml.safe_load(open(args.model_yaml_path))
    model = model_yaml["engine_config"]["model_id"]

    subprocess.run(["aviary", "run", "--model", args.model_yaml_path, "--restart"])

    prompt = (
        "Tell me about the history of the United States of America. Be very "
        "detailed and verbose. Write at least 2000 words."
    )

    wait_for_queue_drain(model_id=model)

    num_concurrent_users_250_tokens = get_max_concurrent_users(
        model=model, prompt=prompt, num_output_tokens=250
    )
    wait_for_queue_drain(model_id=model)
    num_concurrent_users_500_tokens = get_max_concurrent_users(
        model=model, prompt=prompt, num_output_tokens=500
    )
    wait_for_queue_drain(model_id=model)
    num_concurrent_users_1000_tokens = get_max_concurrent_users(
        model=model, prompt=prompt, num_output_tokens=1000
    )
    wait_for_queue_drain(model_id=model)
    num_concurrent_users_2000_tokens = get_max_concurrent_users(
        model=model, prompt=prompt, num_output_tokens=2000
    )

    print(f"250 tokens: {num_concurrent_users_250_tokens}")
    print(f"500 tokens: {num_concurrent_users_500_tokens}")
    print(f"1000 tokens: {num_concurrent_users_1000_tokens}")
    print(f"2000 tokens: {num_concurrent_users_2000_tokens}")
    average_concurrent_users = (
        num_concurrent_users_250_tokens
        + num_concurrent_users_500_tokens
        + num_concurrent_users_1000_tokens
        + num_concurrent_users_2000_tokens
    ) / 4
    print(f"Average concurrent users: {average_concurrent_users}")

    if "/" in model:
        model_name = model
        model_name = model_name.replace("/", "-")

    results = EndpointResults(
        name=f"serve_autoscaling_constants_{model_name}",
        metadata={"average_concurrent_users": average_concurrent_users},
        model_yaml=model_yaml,
        git_commit=os.environ.get("GIT_COMMIT", ""),
    )

    results_dir = args.results_dir
    if results_dir == "":
        results_dir = os.environ.get("AVIARY_RESULTS_DIR", ".")

    results_path = f"{results.timestamp}_serve_autoscaling_constants_{model_name}.json"
    if results_dir != ".":
        results_path = f"{results_dir}/{results_path}"

    with open(
        results_path,
        "w",
    ) as f:
        f.write(results.json())

    if args.s3_bucket == "":
        s3_bucket = os.environ.get("AVIARY_S3_RESULTS_PATH", "")
    else:
        s3_bucket = args.s3_bucket

    if s3_bucket != "":
        print("Uploading to s3")
        upload_to_s3(
            results_path=results_dir,
            s3_path=f"{s3_bucket}/serve_autoscaling_constants/{model_name}",
        )
