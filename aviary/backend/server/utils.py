import os
from typing import List, Union

from aviary.backend.server.models import LLMApp


def parse_args(args: Union[str, LLMApp, List[Union[LLMApp, str]]]) -> List[LLMApp]:
    """Parse the input args and return a standardized list of LLMApp objects

    Supported args format:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    """
    raw_models = []
    if isinstance(args, list):
        raw_models = args
    else:
        raw_models = [args]

    # For each
    models: List[LLMApp] = []
    for raw_model in raw_models:
        if isinstance(raw_model, str):
            parsed_models = _parse_path_args(raw_model)
        else:
            parsed_models = [LLMApp.parse_obj(raw_model)]
        models += parsed_models
    return [model for model in models if model.enabled]


def _parse_path_args(path: str) -> List[LLMApp]:
    assert os.path.exists(
        path
    ), f"Could not load model from {path}, as it does not exist."
    if os.path.isfile(path):
        with open(path, "r") as f:
            return [LLMApp.parse_yaml(f)]
    elif os.path.isdir(path):
        apps = []
        for root, _dirs, files in os.walk(path):
            for p in files:
                if _is_yaml_file(p):
                    with open(os.path.join(root, p), "r") as f:
                        apps.append(LLMApp.parse_yaml(f))
        return apps
    else:
        raise ValueError(
            f"Could not load model from {path}, as it is not a file or directory."
        )


def _is_yaml_file(filename: str) -> bool:
    yaml_exts = [".yml", ".yaml", ".json"]
    for s in yaml_exts:
        if filename.endswith(s):
            return True
    return False
