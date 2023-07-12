"""This file emulates a constant stream of requests."""

import json
import os
import random
import time
from typing import Optional

from locust import HttpUser, constant, task

PROMPTS = [
    "When was George Washington president?",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
    "Explain the difference between Spark and Ray.",
    "Suggest some fun holiday ideas.",
    "Tell a joke.",
    "What is 2+2?",
    "Explain what is machine learning like I am five years old.",
    "Explain what is artifical intelligence.",
    "How do I make fried rice?",
    "What are the most influential punk bands of all time?",
    "What are the best places in the world to visit?",
    "Which Olympics were held in Australia? What years and what cities?",
]

MODELS = [
    # "amazon/LightGPT",
    # "OpenAssistant/falcon-7b-sft-top1-696",
    "OpenAssistant/falcon-40b-sft-top1-560",
    # "RWKV/rwkv-raven-14b",
]

BACKEND_TOKEN = os.getenv("AVIARY_TOKEN", "")


class User(HttpUser):
    wait_time = constant(0)

    def query(self, max_iterations: Optional[int] = None):
        prompt = random.choice(PROMPTS)
        model = random.choice(MODELS)
        model = model.replace("/", "--")
        with self.client.post(
            f"/stream/{model}",
            json={
                "prompt": prompt,
                "parameters": {"max_new_tokens": 512},
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
                    iteration = 0
                    time_since_last_chunk = time.monotonic()
                    for chunk in response.iter_lines(
                        chunk_size=None, decode_unicode=True
                    ):
                        iteration += 1
                        if iteration == max_iterations:
                            # Disconnect request
                            break
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

    @task(10)
    def query_and_consume(self):
        """Emulates a user that consumes an entire request."""
        self.query(max_iterations=None)

    @task(1)
    def query_and_skip(self):
        """Emulates a disconnect while the response is streaming."""
        self.query(max_iterations=2)  # Disconnect after two iterations
