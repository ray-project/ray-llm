import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

from fastapi import Body, FastAPI, Request
from starlette.responses import StreamingResponse

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
    prompt: Annotated[str, Body()],
    request: Request,
    suffix: Annotated[Optional[str], Body()] = None,
    max_tokens: Annotated[int, Body()] = 32,
    temperature: Annotated[float, Body()] = 1.0,
    top_p: Annotated[float, Body()] = 1.0,
    n: Annotated[int, Body()] = 1,
    stream: Annotated[bool, Body()] = False,
    logprobs: Annotated[Optional[int], Body()] = None,
    echo: Annotated[bool, Body()] = False,
    stop: Annotated[Optional[List[str]], Body()] = None,
    presence_penalty: Annotated[float, Body()] = 0.0,
    frequency_penalty: Annotated[float, Body()] = 0.0,
    best_of: Annotated[int, Body()] = 1,
    logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
    user: Annotated[Optional[str], Body()] = None,
):
    model = model.replace("--", "/")
    if stream:

        def gen():
            for word in dummy_generation["generated_text"].split():
                choices = [
                    TextChoice(
                        text=word,
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

                yield Completion(
                    id=model + "-" + str(uuid.uuid4()),
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=choices,
                    usage=usage,
                ).json() + "\n"

        return StreamingResponse(gen(), media_type="text/plain")
    else:
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
    messages: List[Message],
    request: Request,
    temperature: Annotated[float, Body()] = 1.0,
    top_p: Annotated[float, Body()] = 1.0,
    n: Annotated[int, Body()] = 1,
    stream: Annotated[bool, Body()] = False,
    logprobs: Annotated[Optional[int], Body()] = None,
    echo: Annotated[bool, Body()] = False,
    stop: Annotated[Optional[List[str]], Body()] = None,
    presence_penalty: Annotated[float, Body()] = 0.0,
    frequency_penalty: Annotated[float, Body()] = 0.0,
    logit_bias: Annotated[Optional[Dict[str, float]], Body()] = None,
    user: Annotated[Optional[str], Body()] = None,
):
    if stream:

        def gen():
            for word in dummy_generation["generated_text"].split():
                choices: List[MessageChoices] = [
                    MessageChoices(
                        message=Message(role="assistant", content=word),
                        index=0,
                        finish_reason="length",
                    )
                ]
                usage = Usage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

                yield ChatCompletion(
                    id=model + "-" + str(uuid.uuid4()),
                    object="text_completion",
                    created=int(time.time()),
                    model=model,
                    choices=choices,
                    usage=usage,
                ).json() + "\n"

        return StreamingResponse(gen(), media_type="text/plain")
    else:
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
