from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound="Model")
TCompletion = TypeVar("TCompletion", bound="Completion")
TChatCompletion = TypeVar("TChatCompletion", bound="ChatCompletion")


class ModelData(BaseModel):
    id: str
    object: str
    owned_by: str
    permission: List[str]


class Model(BaseModel):
    data: List[ModelData]
    object: str = "list"

    @classmethod
    def list(cls) -> TModel:
        pass


class TextChoice(BaseModel):
    text: str
    index: int
    logprobs: dict
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "Usage":
        return cls(
            prompt_tokens=response["num_input_tokens"],
            completion_tokens=response["num_generated_tokens"],
            total_tokens=response["num_input_tokens"]
            + response["num_generated_tokens"],
        )


class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[TextChoice]
    usage: Usage

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        use_prompt_format: bool = True,
        max_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
    ) -> TCompletion:
        pass


class Message(BaseModel):
    role: str
    content: str


class MessageChoices(BaseModel):
    message: Message
    index: int
    finish_reason: str


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[MessageChoices]
    usage: Usage

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
    ) -> TChatCompletion:
        pass
