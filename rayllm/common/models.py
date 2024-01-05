from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, root_validator, validator

if TYPE_CHECKING:
    from rayllm.backend.server.models import AviaryModelResponse

TModel = TypeVar("TModel", bound="Model")
TCompletion = TypeVar("TCompletion", bound="Completion")
TChatCompletion = TypeVar("TChatCompletion", bound="ChatCompletion")

PROMPT_TRACE_KEY = "+TRACE_"


class PromptFormatDisabledError(ValueError):
    status_code = 404


class ModelData(BaseModel):
    id: str
    object: str
    owned_by: str
    permission: List[str]
    rayllm_metadata: Dict[str, Any]


class Model(BaseModel):
    data: List[ModelData]
    object: str = "list"

    @classmethod
    def list(cls) -> TModel:
        pass


class DeletedModel(BaseModel):
    id: str
    object: str = "model"
    deleted: bool = True


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


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


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


class EmbeddingsData(BaseModel):
    embedding: List[float]
    index: int
    object: str


class EmbeddingsOutput(BaseModel):
    data: List[EmbeddingsData]
    id: str
    object: str
    created: int
    model: str
    usage: Optional[EmbeddingsUsage]


class FunctionCall(BaseModel):
    name: str
    arguments: Optional[str] = None


class ToolCall(BaseModel):
    function: FunctionCall
    type: Literal["function"]
    id: str

    def __str__(self):
        return str(self.dict())


class Function(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Function


class Tool(BaseModel):
    type: Literal["function"]
    function: Function


class Message(BaseModel):
    role: Literal["system", "assistant", "user", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    def __str__(self):
        # if tool_calls is not None, then we are passing a tool message
        # using get attr instead of  just in case the attribute is deleted off of
        # the object
        if getattr(self, "tool_calls", None):
            return str(self.content)
        return str(self.dict())

    @root_validator
    def check_fields(cls, values):
        if values["role"] in ["system", "user"]:
            if not isinstance(values.get("content"), str):
                raise ValueError("content must be a string")
        if values["role"] == "tool":
            if not isinstance(values.get("tool_call_id"), str):
                raise ValueError("tool_call_id must be a str")
            # content should either be a dict with errors or with results
            if not isinstance(values.get("content"), str):
                raise ValueError(
                    "content must be a str with results or errors for " "the tool call"
                )
        if values["role"] == "assistant":
            if values.get("tool_calls"):
                # passing a regular assistant message
                if not isinstance(values.get("tool_calls"), list):
                    raise ValueError("tool_calls must be a list")
                for tool_call in values["tool_calls"]:
                    if not isinstance(tool_call, ToolCall):
                        raise TypeError("Tool calls must be of type ToolCall")
            else:
                # passing a regular assistant message
                if (
                    not isinstance(values.get("content"), str)
                    or values.get("content") == ""
                ):
                    raise ValueError("content must be a string or None")
        return values


class DeltaRole(BaseModel):
    role: Literal["system", "assistant", "user"]

    def __str__(self):
        return self.role


class DeltaContent(BaseModel):
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def __str__(self):
        if self.tool_calls:
            return str(self.tool_calls)
        else:
            return str(self.dict())


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
    tools: Optional[List[Tool]] = None
    tool_choice: Union[Literal["auto", "none"], ToolChoice] = "auto"

    @validator("prompt")
    def check_prompt(cls, value):
        if isinstance(value, list) and not value:
            raise ValueError("Messages cannot be an empty list.")
        return value

    def to_unformatted_string(self) -> str:
        if isinstance(self.prompt, list):
            return ", ".join(str(message.content) for message in self.prompt)
        return self.prompt

    def get_log_str(self):
        prompt_str = self.to_unformatted_string()
        if PROMPT_TRACE_KEY in prompt_str:
            start_idx = prompt_str.find(PROMPT_TRACE_KEY)

            # Grab the prompt key and the next following 30 chars.
            return prompt_str[start_idx : start_idx + len(PROMPT_TRACE_KEY) + 31]
        else:
            return None


class ErrorResponse(BaseModel):
    message: str
    internal_message: str
    code: int
    type: str
    param: Dict[str, Any] = {}


class AbstractPromptFormat(BaseModel):
    class Config:
        extra = "forbid"

    def generate_prompt(self, messages: Union[Prompt, List[Message]]) -> str:
        raise NotImplementedError()


class PromptFormat(AbstractPromptFormat):
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


class DisabledPromptFormat(AbstractPromptFormat):
    def generate_prompt(self, messages: Union[Prompt, List[Message]]) -> str:
        if (
            isinstance(messages, Prompt)
            and isinstance(messages.prompt, str)
            and not messages.use_prompt_format
        ):
            return messages.prompt
        raise PromptFormatDisabledError(
            "This model doesn't support chat completions. Please use the completions "
            "endpoint instead."
        )
