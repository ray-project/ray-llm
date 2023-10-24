from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, root_validator, validator

if TYPE_CHECKING:
    from rayllm.backend.server.models import AviaryModelResponse

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
            response_dict = response.dict()
        else:
            response_dict = response
        return cls(
            prompt_tokens=response_dict["num_input_tokens"] or 0,
            completion_tokens=response_dict["num_generated_tokens"] or 0,
            total_tokens=(response_dict["num_input_tokens"] or 0)
            + (response_dict["num_generated_tokens"] or 0),
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
    parameters: Optional[Union[Dict[str, Any], BaseModel]] = None


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
    system_in_user: bool = False
    add_system_tags_even_if_message_is_empty: bool = False
    strip_whitespace: bool = True

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

    @root_validator
    def check_user_system_in_user(cls, values):
        if values["system_in_user"]:
            assert (
                "{system}" in values["user"]
            ), "If system_in_user=True, user must contain '{system}'"
        return values

    def generate_prompt(self, messages: Union[Prompt, List[Message]]) -> str:
        if isinstance(messages, Prompt):
            if isinstance(messages.prompt, str):
                if not messages.use_prompt_format:
                    return messages.prompt
                new_messages = []
                if self.default_system_message:
                    new_messages.append(
                        Message(role="system", content=self.default_system_message),
                    )
                new_messages.append(
                    Message(role="user", content=messages.prompt),
                )
                messages = new_messages
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

        system_message = None
        if system_message_index != -1:
            system_message = messages.pop(system_message_index)
        elif (
            self.default_system_message or self.add_system_tags_even_if_message_is_empty
        ):
            system_message = Message(role="system", content=self.default_system_message)
        if (
            system_message is not None
            and (
                system_message.content or self.add_system_tags_even_if_message_is_empty
            )
            and not self.system_in_user
        ):
            messages.insert(0, system_message)

        prompt = []
        for message in messages:
            message_content = message.content
            if self.strip_whitespace:
                message_content = message_content.strip()
            if message.role == "system":
                prompt.append(self.system.format(instruction=message_content))
            elif message.role == "user":
                if self.system_in_user:
                    prompt.append(
                        self.user.format(
                            instruction=message_content,
                            system=self.system.format(
                                instruction=system_message.content
                            )
                            if system_message
                            else "",
                        )
                    )
                    system_message = None
                else:
                    prompt.append(self.user.format(instruction=message_content))
            elif message.role == "assistant":
                prompt.append(self.assistant.format(instruction=message_content))
        prompt.append(self.trailing_assistant)
        return "".join(prompt)
