import functools
import json
import os
import subprocess
import time
from typing import Any, Dict

import requests

RESULTS_VERSION = "2023-08-31"


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        end_time - start_time
        ", ".join(
            [repr(arg) for arg in args]
            + [f"{key}={value!r}" for key, value in kwargs.items()]
        )
        return result

    return wrapper


def get_queue_size(model_id: str, endpoints_url: str = None):
    if endpoints_url is None:
        endpoints_url = os.environ.get("AVIARY_URL")
    if "/" in model_id:
        model_id = model_id.replace("/", "--")
    url = endpoints_url + f"/{model_id}/stats"
    response = requests.get(url).json()
    return response["scheduler_stats"]["queue_size"]


@timer
def wait_for_queue_drain(model_id: str, endpoints_url: str = None):
    """Wait for the queue to drain for the given hosted model at the endpoints url."""
    if endpoints_url is None:
        endpoints_url = os.environ.get("AVIARY_URL")
    while True:
        queue_size = get_queue_size(model_id, endpoints_url)
        if queue_size == 0:
            break


class EndpointResults:
    def __init__(
        self,
        name: str,
        metadata: Dict[str, Any] = None,
        git_commit: str = None,
        model_yaml: Dict[str, Any] = None,
    ):
        self.name = name
        self.metadata = metadata
        self.timestamp = int(time.time())
        self.id = f"{self.timestamp}-{name}"
        self.version = RESULTS_VERSION
        self.git_commit = git_commit
        self.model_yaml = model_yaml

    def _get_dict(self):
        data = {
            "version": self.version,
            "id": self.id,
            "name": self.name,
            "model_yaml": self.model_yaml,
        }
        if self.metadata is None:
            data["metadata"] = {}
        else:
            data["metadata"] = self.metadata

        if self.git_commit is None:
            data["git_commit"] = "not reported"
        else:
            data["git_commit"] = self.git_commit

        if self.model_yaml is None:
            data["model_yaml"] = "not reported"
        else:
            data["model_yaml"] = self.model_yaml

        return data

    def json(self):
        data = self._get_dict()
        return json.dumps(data)

    @staticmethod
    def json_multiple_results(*results: "EndpointResults"):
        data = [result._get_dict() for result in results]
        return json.dumps(data)


def upload_to_s3(results_path: str, s3_path: str):
    """Upload the results to s3."""

    command = ["aws", "s3", "sync", results_path, f"{s3_path}/"]
    result = subprocess.run(command)
    if result.returncode == 0:
        print("Files uploaded successfully!")
    else:
        print("An error occurred:")
        print(result.stderr)
