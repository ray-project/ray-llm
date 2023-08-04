import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from aviary.backend.llm.continuous.tokenstream import FinishReason, TokenStream

from .error_handling import ErrorReason

_request_id = -1


def get_request_id() -> int:
    # TODO: more robust request id generation.
    global _request_id
    _request_id += 1
    return _request_id


@pydantic_dataclass
class TGIParams:
    """ "Sampling Parameters for the Text Generation Inference Models.

    Args:
        temperature: Used to control the of text generation. Higher values (e.g., 1.0)
            make the output more diverse and random, while lower values (e.g., 0.2) make
            it more focused and deterministic. Must be greater than or equal to 0.0.
        repetition_penalty: Discourages the model from repeating the same tokens in its
            output. Must be greater than 0.0.
        top_k: Limits the set of tokens considered during text generation. It specifies
            the number of highest probability tokens to keep. Setting it to 0 or a
            negative value means all tokens are considered.
        top_p: Used for nucleus sampling (also known as "top-p" or "penalty" sampling).
            It selects the smallest possible set of tokens whose cumulative probability
            exceeds the value specified (0 < top_p <= 1.0).
        typical_p: Used in conjunction with top_p to set a probability threshold for
            token selection. It's a typical probability value (0 < typical_p <= 1.0) for
            tokens to be included in the output when using nucleus sampling.
        do_sample: Determines whether to use sampling during text generation. If set to
            True, the model will use sampling to select the next token; otherwise, it
            will use greedy decoding.
        stop_sequences: A list of tokens or token IDs that the model should consider as
            stopping conditions during text generation. When one of these tokens is
            generated, the text generation process stops.
        ignore_eos_token: Indicates whether to ignore the end-of-sequence
            token. If set to True, the model will not stop generating text when
            encountering the end-of-sequence token.
        watermark: Determines whether to include a watermark in the
            generated text, probably for identification or tracking purposes.
        seed: Allows you to set a random seed for reproducible text generation. Setting
            it to a specific value will ensure that the same sequence is generated each
            time when using the same seed.
        min_new_tokens: Sets a minimum number of new tokens to be generated in the
            output. This is useful when you want to enforce a certain level of text
            expansion.
        frequency_penalty: Applies a penalty to tokens based on their frequency in
            the input. A higher value penalizes frequently appearing tokens more.
        presence_penalty: Applies a penalty to tokens based on their presence in the
            input. A higher value penalizes tokens that are already present in the
            input more.

    """

    temperature: Optional[float] = Field(1.0, ge=0.0)
    repetition_penalty: float = Field(1.0, gt=0.0)
    top_k: Optional[int] = Field(0, ge=0)
    top_p: Optional[float] = Field(1.0, gt=0, le=1.0)
    typical_p: Optional[float] = Field(1.0, gt=0, le=1.0)
    do_sample: bool = True
    stop_sequences: List[Union[str, int, List[int]]] = field(default_factory=list)
    ignore_eos_token: bool = False
    watermark: bool = False
    seed: Optional[int] = Field(None, ge=0)
    # Aviary params
    # Force at least one token to be generated.
    min_new_tokens: int = Field(1, gt=0)
    # OpenAI repetition penalties
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        if not self.temperature:
            # if temperature is 0, do not do any sampling.
            self.temperature = 1.0
            self.top_k = 0
            self.top_p = 1.0
            self.typical_p = 1.0
            self.do_sample = False
        else:
            # Set None values to noop values since conversion to TGI pb2
            # objects will treat None as 0.0
            if not self.typical_p:
                self.typical_p = 1.0
            if not self.top_p:
                self.top_p = 1.0
            if not self.top_k:
                self.top_k = 0


@dataclass
class Request:
    inputs: str
    truncate: int
    max_new_tokens: int
    input_tokens: int
    params: TGIParams
    id: Optional[int] = None

    @property
    def batch_id(self) -> int:
        return self.id

    def __post_init__(self):
        if self.id is None:
            self.id = get_request_id()

    @property
    def truncated_input_tokens(self) -> int:
        return min(self.truncate, self.input_tokens)


@dataclass
class InferenceTask:
    request: Request

    def __post_init__(self):
        self._output_stream = TokenStream(self.id)
        self._submit_time_s = time.time()

    def mark_as_invalid(self, reason: ErrorReason):
        self.mark_as_finished(FinishReason.ERROR, reason)

    def mark_as_finished(
        self, finish_reason: FinishReason, error_reason: Optional[ErrorReason] = None
    ):
        self.output_stream.end(finish_reason, error_reason=error_reason)

    @property
    def id(self) -> int:
        return self.request.id

    @property
    def output_stream(self) -> TokenStream:
        return self._output_stream

    @property
    def is_finished(self):
        return self.output_stream.is_finished

    @property
    def total_tokens(self) -> int:
        return self.request.truncated_input_tokens + self.request.max_new_tokens

    @property
    def input_length(self) -> int:
        return self.request.truncated_input_tokens

    @property
    def actual_total_tokens(self) -> int:
        return self.request.input_tokens + self.output_stream.num_tokens

    @property
    def generated_tokens(self) -> int:
        return self.output_stream.num_tokens

    @property
    def gen_length(self) -> int:
        return max(
            0,
            self.request.max_new_tokens - self.output_stream.num_tokens,
        )

    @property
    def decode_cost(self) -> int:
        return self.request.max_new_tokens

    @property
    def total_cost(self) -> int:
        return self.decode_cost + self.input_cost

    @property
    def input_cost(self) -> int:
        return self.input_length


__all__ = ["Request", "TGIParams", "InferenceTask"]
