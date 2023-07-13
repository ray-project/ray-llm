import time
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI

from aviary.common.models import (
    ChatCompletion,
    Completion,
    Message,
    MessageChoices,
    Model,
    ModelData,
    TextChoice,
    Usage,
)

app = FastAPI()

dummy_generation = {
    "generated_text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
}

all_models = ["hf-internal-testing--tiny-random-gpt2"]


@app.post("/query/{model}")
def query(model: str) -> Dict[str, Any]:
    model = model.replace("--", "/")
    return {
        "generated_text": dummy_generation,
        "total_time": 1.0,
        "num_total_tokens": 42,
    }


@app.post("/query/batch/{model}")
def batch_query(model: str) -> List[Dict[str, Any]]:
    model = model.replace("--", "/")
    return [
        {"generated_text": dummy_generation},
        {"generated_text": dummy_generation},
    ]


@app.get("/metadata/{model}")
def metadata(model) -> Dict[str, Dict[str, Any]]:
    return {
        "metadata": {
            "model_config": {
                "model_description": "dummy model",
                "model_url": "dummy_url",
            }
        }
    }


@app.get("/models")
def models_v0() -> List[str]:
    return [model.replace("--", "/") for model in all_models]


@app.get("/v1/models")
def models() -> Model:
    model_ids = ["dummy_model_1", "dummy_model_2"]
    model_list = []
    for model_id in model_ids:
        model_list.append(
            ModelData(
                id=model_id,
                object="model",
                owned_by="organization-owner",
                permission=[],
            )
        )
    return Model(data=model_list)


@app.get("/v1/models/{model}")
def model_data(model: str) -> ModelData:
    return ModelData(
        id=model, object="model", owned_by="organization-owner", permission=[]
    )


@app.post("/v1/completions/{model}", response_model=Completion)
def completions(
    model: str,
) -> Completion:
    model = model.replace("--", "/")
    choices = [
        TextChoice(
            text=dummy_generation["generated_text"],
            index=0,
            logprobs={},
            finish_reason="length",
        )
    ]
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    return Completion(
        id=model + "-" + str(uuid.uuid4()),
        object="text_completion",
        created=int(time.time()),
        model=model,
        choices=choices,
        usage=usage,
    )


@app.post("/v1/chat/completions/{model}", response_model=ChatCompletion)
def chat(
    model: str,
) -> ChatCompletion:
    choices: List[MessageChoices] = [
        MessageChoices(
            message=Message(
                role="assistant", content=dummy_generation["generated_text"]
            ),
            index=0,
            finish_reason="length",
        )
    ]
    usage = Usage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    return ChatCompletion(
        id=model + "-" + str(uuid.uuid4()),
        object="text_completion",
        created=int(time.time()),
        model=model,
        choices=choices,
        usage=usage,
    )
