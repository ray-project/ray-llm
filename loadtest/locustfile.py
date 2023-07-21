import json
import os
import random

import gevent
from gevent.pool import Group
from locust import HttpUser, LoadTestShape, TaskSet, between, task

prompts = [
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

SELECTION_DICT = {
    "Fast": [
        "amazon/LightGPT",
        "stabilityai/stablelm-tuned-alpha-7b",
        "mosaicml/mpt-7b-chat",
    ],
    "Strong": [
        "CarperAI/stable-vicuna-13b-delta",
        "OpenAssistant/oasst-sft-7-llama-30b-xor",
        "mosaicml/mpt-7b-chat",
    ],
    "Variants": [
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-chat",
        "mosaicml/mpt-7b-storywriter",
    ],
    "Random": [
        "amazon/LightGPT",
        "CarperAI/stable-vicuna-13b-delta",
        "databricks/dolly-v2-12b",
        "h2oai/h2ogpt-oasst1-512-12b",
        "lmsys/vicuna-13b-delta-v1.1",
        "mosaicml/mpt-7b-chat",
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-storywriter",
        "OpenAssistant/oasst-sft-7-llama-30b-xor",
        # "RWKV/rwkv-raven-14b",
        "stabilityai/stablelm-tuned-alpha-7b",
    ],
}

backend_token = os.getenv("AVIARY_TOKEN", "")
bearer = f"Bearer {backend_token}"


class UserTasks(TaskSet):
    @task
    def query_model(self):
        prompt = random.choice(prompts)

        models = random.choice(list(SELECTION_DICT.values()))
        if len(models) > 3:
            models = [random.choice(models) for _ in range(3)]
        models = ["mosaicml/mpt-7b-chat"]
        models = [model.replace("/", "--") for model in models]

        def func(model):
            with self.client.post(
                f"/{model}/stream",
                json={"prompt": prompt},
                headers={"Authorization": bearer},
                timeout=120,
                catch_response=True,
                stream=True,
            ) as response:
                for chunk in response.iter_lines(chunk_size=None, decode_unicode=True):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    data = json.loads(chunk)
                    if data.get("error"):
                        raise RuntimeError(data["error"], response=response)

        group = Group()
        jobs = [group.spawn(func, model=model) for model in models]
        gevent.wait(jobs)


class WebsiteUser(HttpUser):
    #    wait_time = between(5, 60)
    wait_time = between(15, 30)
    tasks = [UserTasks]


class StagesShape(LoadTestShape):
    """
    A simply load test shape class that has different user and spawn_rate at
    different stages.

    Keyword arguments:

        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage

        stop_at_end -- Can be set to stop once all stages have run.
    """

    multiplier = 5
    time_multiplier = 0.5
    stages = [
        {"duration": 100 * time_multiplier, "users": 10, "spawn_rate": 10},
        {
            "duration": 800 * time_multiplier,
            "users": int(100 * multiplier),
            "spawn_rate": 0.125 * multiplier / time_multiplier,
        },
        {
            "duration": 1800 * time_multiplier,
            "users": int(100 * multiplier),
            "spawn_rate": 100 * multiplier / time_multiplier,
        },
        {
            "duration": 3500 * time_multiplier,
            "users": int(10 * multiplier),
            "spawn_rate": 0.125 * multiplier / time_multiplier,
        },
        {
            "duration": 3600 * time_multiplier,
            "users": int(10 * multiplier),
            "spawn_rate": 10,
        },
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None


# class DoubleWave(LoadTestShape):
#     """
#     A shape to imitate some specific user behaviour. In this example, midday
#     and evening meal times. First peak of users appear at time_limit/3 and
#     second peak appears at 2*time_limit/3

#     Settings:
#         min_users -- minimum users
#         peak_one_users -- users in first peak
#         peak_two_users -- users in second peak
#         time_limit -- total length of test
#     """

#     min_users = 10
#     peak_one_users = 80
#     peak_two_users = 60
#     time_limit = 3600

#     def tick(self):
#         run_time = round(self.get_run_time())

#         if run_time < self.time_limit:
#             user_count = (
#                 (self.peak_one_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 5) ** 2)
#                 + (self.peak_two_users - self.min_users)
#                 * math.e ** -(((run_time / (self.time_limit / 10 * 2 / 3)) - 10) ** 2)
#                 + self.min_users
#             )
#             return (round(user_count), round(user_count))
#         else:
#             return None
