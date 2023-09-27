import json
import os
import random
from typing import Any, Dict, List, Union

from locust import HttpUser, TaskSet, between, task


def create_completions_request(
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
                prompt = random.choice(prompts)
                body = {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": num_output_tokens,
                }
                body.update(sampling_params or {})
                try:
                    with self.client.post(
                        "/v1/completions", json=body, stream=False, timeout=180
                    ) as response:
                        response.raise_for_status()
                        for chunk in response.iter_lines(chunk_size=None):
                            chunk = chunk.strip()
                            if not chunk:
                                continue
                            data = json.loads(chunk)
                            assert (
                                data["usage"]["completion_tokens"] == num_output_tokens
                            ), f"num_output_tokens = {data['usage']['completion_tokens']}. Should be {num_output_tokens}"
                            if data.get("error"):
                                raise RuntimeError(data["error"])
                except Exception as e:
                    print(f"Warning Or Error: {e}")

            func(model=model)

    return LaunchEndpointRequest


def create_user(
    requests: Union[TaskSet, List[TaskSet]],
    endpoint_address: str = None,
):
    """Create a Locust User class that will send requests to an endpoint.

    Args:
        requests: A list of TaskSet classes that send requests to the endpoint.

    Returns:
        A Locust User class that will send requests to the endpoint.
    """
    if endpoint_address is None:
        endpoint_address = os.environ.get("AVIARY_URL")

    if not isinstance(requests, list):
        requests = [requests]

    class EndpointsUser(HttpUser):
        """Launch requests to the endpoints at the AVIARY_URL Address."""

        wait_time = between(0, 0.1)
        tasks = requests
        host = endpoint_address

    return EndpointsUser
