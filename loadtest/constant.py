"""This file emulates a constant stream of requests."""

import json
import os
import random
import time

from locust import HttpUser, constant, task

MODELS = [
    # "amazon/LightGPT",
    # "OpenAssistant/falcon-7b-sft-top1-696",
    "OpenAssistant/falcon-40b-sft-top1-560",
    # "mosaicml/mpt-30b-chat",
]

BACKEND_TOKEN = os.getenv("AVIARY_TOKEN", "")


class ConstantUser(HttpUser):
    """User sends the same prompt, which usually generates a short response.

    Use this test to check Aviary's throughput with a relatively constant
    workload. The prompt is the same each time, and the response is usually
    pretty short.
    """

    PROMPT = "<|assistant|>" * 512  # * 900

    wait_time = constant(0)

    @task
    def query(self):
        model = random.choice(MODELS)
        model = model.replace("/", "--")
        with self.client.post(
            f"/stream/{model}",
            json={
                "prompt": self.PROMPT,
                "use_prompt_format": False,
                "parameters": {
                    "do_sample": False,
                    "stopping_sequences": [],
                    "min_new_tokens": 128,
                    "max_new_tokens": 128,
                },
            },
            headers={"Authorization": f"Bearer {BACKEND_TOKEN}"},
            timeout=120,
            catch_response=True,
            stream=True,
        ) as response:
            try:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Got non-200 response code: {response.status_code}."
                    )
                else:
                    chunks = []
                    time_since_last_chunk = time.monotonic()
                    for chunk in response.iter_lines(
                        chunk_size=None, decode_unicode=True
                    ):
                        chunk = chunk.strip()
                        if chunk:
                            if time.monotonic() - time_since_last_chunk > 120:
                                raise RuntimeError(
                                    "* Chunk timeout." f"\n* Chunks so far: {chunks}"
                                )
                            time_since_last_chunk = time.monotonic()
                            chunks.append(chunk)
                            data = json.loads(chunk)
                            if data.get("error"):
                                raise RuntimeError(
                                    f"* Data chunk contained an error: {data}"
                                    f"\n* Chunks so far: {chunks}"
                                )
            except Exception as e:
                response.failure(f"Exception: {e}")
            response.success()
        return response
