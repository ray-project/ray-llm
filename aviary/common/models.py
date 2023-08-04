from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, validator

if TYPE_CHECKING:
    from aviary.backend.server.models import AviaryModelResponse

TModel = TypeVar("TModel", bound="Model")
TCompletion = TypeVar("TCompletion", bound="Completion")
TChatCompletion = TypeVar("TChatCompletion", bound="ChatCompletion")


class ModelData(BaseModel):
    id: str
    object: str
    owned_by: str
    permission: List[str]
    aviary_metadata: Dict[str, Any]


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
    finish_reason: Optional[str]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_response(
        cls, response: Union["AviaryModelResponse", Dict[str, Any]]
    ) -> "Usage":
        if isinstance(response, BaseModel):
            response = response.dict()
        return cls(
            prompt_tokens=response["num_input_tokens"] or 0,
            completion_tokens=response["num_generated_tokens"] or 0,
            total_tokens=(response["num_input_tokens"] or 0)
            + (response["num_generated_tokens"] or 0),
        )


class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[TextChoice]
    usage: Optional[Usage]

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        use_prompt_format: bool = True,
        max_tokens: Optional[int] = 16,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> TCompletion:
        pass


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content


class DeltaRole(BaseModel):
    role: Literal["system", "assistant", "user"]

    def __str__(self):
        return self.role


class DeltaContent(BaseModel):
    content: str

    def __str__(self):
        return self.content


class DeltaEOS(BaseModel):
    class Config:
        extra = "forbid"


class MessageChoices(BaseModel):
    message: Message
    index: int
    finish_reason: str


class DeltaChoices(BaseModel):
    delta: Union[DeltaRole, DeltaContent, DeltaEOS]
    index: int
    finish_reason: Optional[str]


class ChatCompletion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Union[MessageChoices, DeltaChoices]]
    usage: Optional[Usage]

    @classmethod
    def create(
        cls,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        frequency_penalty: float = 0.0,
        top_k: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = False,
        seed: Optional[int] = None,
    ) -> TChatCompletion:
        pass


class Prompt(BaseModel):
    prompt: Union[str, List[Message]]
    use_prompt_format: bool = True
    parameters: Optional[Dict[str, Any]] = None
    stopping_sequences: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    message: str
    internal_message: str
    code: int
    type: str
    param: Dict[str, Any] = {}


class PromptFormat(BaseModel):
    system: str
    assistant: str
    trailing_assistant: str
    user: str

    default_system_message: str = ""

    @validator("system")
    def check_system(cls, value):
        assert value and (
            "{instruction}" in value
        ), "system must be a string containing '{instruction}'"
        return value

    @validator("assistant")
    def check_assistant(cls, value):
        assert (
            value and "{instruction}" in value
        ), "assistant must be a string containing '{instruction}'"
        return value

    @validator("user")
    def check_user(cls, value):
        assert value and (
            "{instruction}" in value
        ), "user must be a string containing '{instruction}'"
        return value

    def generate_prompt(self, messages: Union[Prompt, List[Message]]) -> str:
        if isinstance(messages, Prompt):
            if isinstance(messages.prompt, str):
                if not messages.use_prompt_format:
                    return messages.prompt
                messages = [
                    Message(role="system", content=self.default_system_message),
                    Message(role="user", content=messages.prompt),
                ]
            else:
                messages = messages.prompt

        # Get system message
        system_message_index = -1
        for i, message in enumerate(messages):
            if message.role == "system":
                if system_message_index == -1:
                    system_message_index = i
                else:
                    raise ValueError("Only one system message can be specified.")

        if system_message_index != -1:
            system_message = messages.pop(system_message_index)
        else:
            system_message = Message(role="system", content=self.default_system_message)
        messages.insert(0, system_message)

        prompt = []
        for message in messages:
            if message.role == "system":
                prompt.append(self.system.format(instruction=message.content))
            elif message.role == "assistant":
                prompt.append(self.assistant.format(instruction=message.content))
            elif message.role == "user":
                prompt.append(self.user.format(instruction=message.content))
        prompt.append(self.trailing_assistant)
        return "".join(prompt)
