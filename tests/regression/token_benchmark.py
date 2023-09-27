import argparse
import copy
import json
import os
import pathlib
import random
import subprocess
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from locust import TaskSet, task
from regression_test_runner import RegressionTestBase
from transformers import AutoTokenizer
from users import create_user
from utils import EndpointResults, upload_to_s3, wait_for_queue_drain

from aviary.conf import ENV_VARS_TO_PROPAGATE  # noqa

# Create a lock
lock = threading.Lock()


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


# Locust doesn't provide an easy way to retrieve custom stats from requests/users
# so instead we write them to this global stats dict
TOKEN_STATS = defaultdict(list)


def create_completions_request_w_stats(
    model: str,
    prompts: List[str],
    num_output_tokens: int,
    sampling_params: Dict[str, Any] = None,
) -> TaskSet:
    """Create a locust task class that sends a request to the completions endpoint.

    Args:
        model: The model to use.
        prompts: Prompts to uniformly sample from to send with each request.
        num_output_tokens: The number of tokens to generate per request.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
            route.

    Returns:
        A locust task class that sends a request to the completions endpoint.
    """

    class LaunchEndpointRequest(TaskSet):
        @task
        def query_model(self):
            def func(model):
                first_token_recieved = False
                prompt, prompt_len = random.choice(prompts)

                message = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt},
                ]
                body = {
                    "model": model,
                    "messages": message,
                    "stream": True,
                    "max_tokens": num_output_tokens,
                }
                body.update(sampling_params or {})
                time_to_next_token = []
                tokens_recieved = 0
                try:
                    start_time = time.time()
                    curr_time = time.time()
                    with self.client.post(
                        "/v1/chat/completions", json=body, stream=True, timeout=180
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_lines(chunk_size=None):
                            chunk = chunk.strip()
                            if not chunk:
                                continue
                            stem = "data: "
                            chunk = chunk[len(stem) :]
                            if chunk == b"[DONE]":
                                total_request_time = time.time() - start_time
                                continue
                            tokens_recieved += 1
                            data = json.loads(chunk)

                            if not first_token_recieved:
                                ttft = time.time() - curr_time
                                first_token_recieved = True
                            else:
                                time_to_next_token.append(time.time() - curr_time)
                            curr_time = time.time()

                            if "error" in data:
                                raise RuntimeError(data["error"]["message"])

                    global TOKEN_STATS
                    throughput = tokens_recieved + prompt_len / (total_request_time)
                    with lock:
                        # write to global stats
                        TOKEN_STATS["token_lat_s"].extend(time_to_next_token)
                        TOKEN_STATS["user_ttft_s"].append(ttft)
                        TOKEN_STATS["user_total_request_time_s"].append(
                            total_request_time
                        )
                        TOKEN_STATS["user_throughput_token_per_s"].append(throughput)
                        TOKEN_STATS["number_tokens_processed"].append(
                            tokens_recieved + prompt_len
                        )

                except Exception as e:
                    print(f"Warning Or Error: {e}")

            func(model=model)

    return LaunchEndpointRequest


def generate_prompts(tokenizer, prompt_length, num_prompts):
    """Generate a list of prompts to use for the benchmark.

    Args:
        tokenizer: The tokenizer to use to tokenize the prompts.
        prompt_length: The length of the prompts to generate.
        num_prompts: The number of prompts to generate.

    Returns:
        A list of prompts to use for the benchmark.
    """

    def gen_prompt_ids(length):
        return [random.randint(10, 50000) for _ in range(length)]

    prompts_as_ids = list(
        map(
            lambda prompt_len: gen_prompt_ids(prompt_len),
            [prompt_length for _ in range(num_prompts)],
        )
    )
    decoded = [
        (tokenizer.decode(prompt_as_ids), len(prompt_as_ids))
        for prompt_as_ids in prompts_as_ids
    ]

    return decoded


def reset_token_stats():
    """Reset the global token stats."""
    global TOKEN_STATS
    TOKEN_STATS = defaultdict(list)


def get_token_throughput_latencies(
    model: str,
    prompts: List[Tuple[str, int]],
    num_output_tokens: int,
    sampling_params: Optional[Dict[str, Any]] = None,
    num_users: int = 500,
    test_runtime_s=90,
) -> int:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The model to use.
        prompts: Prompts to uniformly sample from to send with each request and the number of tokens in each prompt.
        num_output_tokens: The number of tokens to generate per request.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        num_users: The number of concurrent users that will send requests. Increase this to increase the amount of
            load and vice versa.
        test_runtime_s: The amount of time to run the test for.


    """
    request_cls = create_completions_request_w_stats(
        model=model,
        prompts=prompts,
        num_output_tokens=num_output_tokens,
        sampling_params=sampling_params,
    )
    user_cls = create_user(requests=request_cls)
    test_runner = RegressionTestBase(user_cls)

    test_runner.set_num_users(num_users=num_users, spawn_rate=100)

    start_time = time.time()

    while time.time() - start_time < test_runtime_s:
        time.sleep(0.0001)

    test_runner.shutdown()
    stats = copy.deepcopy(TOKEN_STATS)
    reset_token_stats()

    ret = {}

    for k, v in stats.items():
        if k == "number_tokens_processed":
            continue
        v = np.array(v)
        p25 = np.percentile(v, 25)
        p50 = np.percentile(v, 50)
        p75 = np.percentile(v, 75)
        p90 = np.percentile(v, 90)
        p95 = np.percentile(v, 95)
        p99 = np.percentile(v, 99)

        ret[f"{k}_p25"] = p25
        ret[f"{k}_p50"] = p50
        ret[f"{k}_p75"] = p75
        ret[f"{k}_p90"] = p90
        ret[f"{k}_p95"] = p95
        ret[f"{k}_p99"] = p99

    ret["throughput_token_per_s"] = (
        np.sum(stats["number_tokens_processed"]) / test_runtime_s
    )

    return ret


def run_test(
    model: str,
    model_yaml: Dict[str, Any],
    num_input_tokens: int,
    num_output_tokens: int,
    results_dir: str,
    s3_bucket: str,
    git_commit: str,
    test_runtime_s=90,
    num_users=500,
):
    """Run the get_token_throughput_latencies test for the given model and num input/output tokens.

    Args:
        model: The model to use.
        model_yaml: The model yaml that was used to start the model.
        num_input_tokens: The number of tokens to send in the prompt for the request.
        num_output_tokens: The number of tokens to generate per request.
        results_dir: The directory to save the results to.
        s3_bucket: The s3 bucket to save the results to.
        git_commit: The git commit of the model.
        test_runtime_s: The amount of time to run the test for.
        num_users: The number of concurrent users that will send requests.

    """
    key = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model, token=key)

    prompts = generate_prompts(
        tokenizer, prompt_length=num_input_tokens, num_prompts=500
    )

    wait_for_queue_drain(model_id=model)

    token_stats = get_token_throughput_latencies(
        model=model,
        prompts=prompts,
        num_output_tokens=num_output_tokens,
        test_runtime_s=test_runtime_s,
        num_users=num_users,
    )

    if "/" in model:
        model_name = model
        model_name = model_name.replace("/", "-")

    metadata = {
        "num_input_tokens": num_input_tokens,
        "num_output_tokens": num_output_tokens,
        "num_users": num_users,
    }
    metadata.update(token_stats)
    print(metadata)

    results = EndpointResults(
        name=f"throughput_latency_benchmarks_{model_name}",
        model_yaml=model_yaml,
        metadata=metadata,
        git_commit=git_commit,
    )

    tp = model_yaml["scaling_config"]["num_workers"]

    results_path = (
        f"{results.timestamp}_throughput_latency_benchmarks_{model_name}_tp_{tp}"
        f"_io_{num_input_tokens}_{num_output_tokens}.json"
    )
    if results_dir != ".":
        results_path = f"{results_dir}/{results_path}"

    with open(
        results_path,
        "w",
    ) as f:
        f.write(results.json())

    if s3_bucket != "":
        print("Uploading to s3")
        upload_to_s3(
            results_path=results_dir,
            s3_path=f"{s3_bucket}/throughput_latency_benchmarks/{model_name}",
        )
    return metadata


if __name__ == "__main__":
    args = parser.parse_args()

    path = pathlib.Path(args.model_yaml_path)
    assert path.exists()

    model_yaml = yaml.safe_load(open(args.model_yaml_path))
    model = model_yaml["engine_config"]["model_id"]

    results_dir = args.results_dir
    if results_dir == "":
        results_dir = os.environ.get("AVIARY_RESULTS_DIR", ".")

    if args.s3_bucket == "":
        s3_bucket = os.environ.get("AVIARY_S3_RESULTS_PATH", "")
    else:
        s3_bucket = args.s3_bucket

    git_commit = os.environ.get("GIT_COMMIT", "")

    subprocess.run(["aviary", "run", "--model", args.model_yaml_path, "--restart"])

    test_configurations = [
        {"num_input_tokens": 512, "num_output_tokens": 128, "num_users": 200},
        {"num_input_tokens": 128, "num_output_tokens": 512, "num_users": 200},
        {"num_input_tokens": 2000, "num_output_tokens": 2000, "num_users": 200},
        {"num_input_tokens": 2, "num_output_tokens": 512, "num_users": 1},
    ]

    results = {}

    for cfg in test_configurations:
        result = run_test(
            model=model,
            model_yaml=model_yaml,
            test_runtime_s=90,
            results_dir=results_dir,
            s3_bucket=s3_bucket,
            git_commit=git_commit,
            **cfg,
        )
        results[f"{cfg['num_input_tokens']}_{cfg['num_output_tokens']}"] = result

    print(
        f"RESULTS FOR THROUGHPUT AND LATENCIES FOR {model} tp={model_yaml['scaling_config']['num_workers']}"
    )
    for k, v in results.items():
        print(f"i/o tokens {k}")
        print(v)
