import time

import gradio as gr

from aviary.common.constants import (
    G5_COST_PER_S_IN_DOLLARS,
    NUM_LLM_OPTIONS,
    PROJECT_NAME,
)
from aviary.frontend.mongo_logger import MongoLogger
from aviary.frontend.mongo_secrets import get_mongo_secret_url

LOGGER = None

MONGODB_URL = get_mongo_secret_url()
if MONGODB_URL:
    LOGGER = MongoLogger(url=MONGODB_URL, project_name=PROJECT_NAME)
else:
    print("No MongoDB logger defined, will default to the CSVLogger")
    LOGGER = gr.CSVLogger()


DEFAULT_STATS = t = """
        | <!-- --> | <!-- --> |
        |---|---|
        | Latency [s] | - |
        | Cost [$] | - |
        | Tokens (i/o) | - |
        | Per 1K Tokens [$] | - |
"""


def gen_stats(dictionary):
    cost_per_k = (
        dictionary["total_time"]
        * G5_COST_PER_S_IN_DOLLARS
        / dictionary["num_total_tokens"]
        * 1000
    )

    return f"""
            | <!-- --> | <!-- --> |
            |---|---|
            | Lat [s] | {dictionary['total_time']:.1f} |
            | Cost [$] | {dictionary['total_time'] * G5_COST_PER_S_IN_DOLLARS:.4f} |
            | Tokens (i/o) | {dictionary['num_total_tokens']:.1f} |
            | Per 1K Tok [$] | {cost_per_k:.4f} |
    """


def blank():
    return ""


def select_button(button):
    return button, gr.Button.update(variant="primary")


def deactivate_buttons():
    return [gr.Button.update(interactive=False)] * NUM_LLM_OPTIONS


def unset_buttons():
    return [gr.Button.update(variant="secondary", interactive=True)] * NUM_LLM_OPTIONS


def paused_logger(*args):
    time.sleep(1)
    LOGGER.flag(*args)


def log_flags(*args):
    LOGGER.flag(args)


THEME = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
).set(
    border_color_accent="blue",
    shadow_spread="20",
    shadow_spread_dark="0",
    button_primary_background_fill="*primary_200",
    button_primary_background_fill_dark="*primary_700",
    button_primary_border_color_dark="*primary_600",
)
