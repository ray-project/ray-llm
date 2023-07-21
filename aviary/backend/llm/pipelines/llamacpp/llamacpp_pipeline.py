import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import torch

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Response

from ...initializers.llamacpp import LlamaCppInitializer, LlamaCppTokenizer
from .._base import StreamingPipeline
from ..utils import decode_stopping_sequences_where_needed

if TYPE_CHECKING:
    from llama_cpp import Llama, LogitsProcessorList, StoppingCriteriaList

logger = get_logger(__name__)


class LlamaCppPipeline(StreamingPipeline):
    """Text generation pipeline using llama.cpp.

    May not support all features."""

    def __init__(
        self,
        model: "Llama",
        tokenizer: LlamaCppTokenizer,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> None:
        from llama_cpp import Llama

        if not isinstance(model, Llama):
            raise TypeError("Model must be an instance of llama_cpp.Llama.")
        self.model = model
        self.kwargs = kwargs
        self.tokenizer = tokenizer
        self.device = device

    def _get_logits_processors(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> "LogitsProcessorList":
        from llama_cpp import LogitsProcessorList

        from aviary.backend.llm.pipelines.llamacpp.processors import (
            LlamaCppMinNewTokensLengthLogitsProcessor,
        )

        lst = []

        if "min_new_tokens" in generate_kwargs:
            lst.append(
                LlamaCppMinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=len(model_inputs["tokenized_inputs"]),
                    min_new_tokens=generate_kwargs.pop("min_new_tokens", 4),
                    eos_token_id=self.model.token_eos(),
                )
            )

        return LogitsProcessorList(lst)

    def _get_stopping_criteria(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> "StoppingCriteriaList":
        from llama_cpp import StoppingCriteriaList

        from aviary.backend.llm.pipelines.llamacpp.processors import (
            LlamaMaxTimeCriteria,
        )

        lst = []

        timeout_s = generate_kwargs.pop("timeout_s", None)
        start_timestamp = generate_kwargs.pop("start_timestamp", None)
        if timeout_s is not None and start_timestamp is not None:
            lst.append(LlamaMaxTimeCriteria(timeout_s, start_timestamp))

        return StoppingCriteriaList(lst)

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Dict[str, Any]:
        generate_kwargs = generate_kwargs.copy()
        generate_kwargs.setdefault("echo", False)
        stopping_sequences = generate_kwargs.pop("stopping_sequences")
        stopping_sequences = decode_stopping_sequences_where_needed(
            self.tokenizer, stopping_sequences
        )
        generate_kwargs.setdefault("stop", stopping_sequences)
        generate_kwargs["logits_processor"] = self._get_logits_processors(
            generate_kwargs, model_inputs=model_inputs
        )
        generate_kwargs["stopping_criteria"] = self._get_stopping_criteria(
            generate_kwargs, model_inputs=model_inputs
        )
        return generate_kwargs

    def __call__(self, inputs: List[str], **kwargs) -> List[Response]:
        tokenized_inputs = self.tokenizer.encode(inputs[0])
        kwargs = self._add_default_generate_kwargs(
            kwargs,
            model_inputs={"inputs": inputs, "tokenized_inputs": tokenized_inputs},
        )

        logger.info(f"Forward params: {kwargs}, model_inputs {inputs}")
        responses = []
        for input in inputs:
            st = time.monotonic()
            output = self.model(input, **kwargs)
            gen_time = time.monotonic() - st
            text = output["choices"][0]["text"].replace("\u200b", "").strip()
            responses.append(
                Response(
                    generated_text=text,
                    num_generated_tokens=output["usage"]["completion_tokens"],
                    num_input_tokens=output["usage"]["prompt_tokens"],
                    num_generated_tokens_batch=output["usage"]["completion_tokens"],
                    num_input_tokens_batch=output["usage"]["prompt_tokens"],
                    preprocessing_time=None,
                    postprocessing_time=None,
                    generation_time=gen_time,
                )
            )
        return responses

    def stream(
        self,
        inputs: List[str],
        **kwargs,
    ) -> Iterator[torch.LongTensor]:
        tokenized_inputs = self.tokenizer.encode(inputs[0])
        kwargs = self._add_default_generate_kwargs(
            kwargs,
            model_inputs={"inputs": inputs, "tokenized_inputs": tokenized_inputs},
        )

        logger.info(f"Forward params: {kwargs}, model_inputs {inputs}")
        first_token_done = False
        for input in inputs:
            for output in self.model(input, stream=True, **kwargs):
                st = time.monotonic()
                gen_time = time.monotonic() - st
                text = output["choices"][0]["text"].replace("\u200b", "")
                if not first_token_done:
                    text = text.lstrip()
                    first_token_done = True
                yield [
                    Response(
                        generated_text=text,
                        num_generated_tokens=1,
                        num_input_tokens=len(tokenized_inputs),
                        num_generated_tokens_batch=1,
                        num_input_tokens_batch=len(tokenized_inputs),
                        preprocessing_time=None,
                        postprocessing_time=None,
                        generation_time=gen_time,
                    )
                ]

    @classmethod
    def from_initializer(
        cls,
        initializer: "LlamaCppInitializer",
        model_id: str,
        device: Optional[Union[str, int, torch.device]] = None,
        **kwargs,
    ) -> "LlamaCppPipeline":
        assert isinstance(initializer, LlamaCppInitializer)
        model, tokenizer = initializer.load(model_id)
        logger.info(f"Model: {model}")
        return cls(
            model,
            tokenizer,
            device=device,
            **kwargs,
        )
