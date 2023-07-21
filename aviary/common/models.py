from typing import Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, validator

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
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content


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


class Prompt(BaseModel):
    prompt: Union[str, List[Message]]
    use_prompt_format: bool = True
    parameters: Optional[Dict[str, Any]] = None
    stopping_sequences: Optional[List[str]] = None


class PromptFormat(BaseModel):
    system: str
    assistant: str
    trailing_assistant: str
    user: str

    default_system_message: str = ""

    @validator("system")
    def check_system(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "system must be empty string or string containing '{instruction}'"
        return value

    @validator("assistant")
    def check_assistant(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "assistant must be empty string or string containing '{instruction}'"
        return value

    @validator("user")
    def check_user(cls, value):
        if value:
            assert (
                "{instruction}" in value
            ), "user must be empty string or string containing '{instruction}'"
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
