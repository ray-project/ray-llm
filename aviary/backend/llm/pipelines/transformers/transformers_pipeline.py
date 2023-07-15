import time
import warnings
from collections import UserDict
from typing import Any, Dict, Iterator, List, Optional, Union
from unittest.mock import patch

import torch
from transformers import (
    LogitsProcessorList,
    MaxTimeCriteria,
    MinNewTokensLengthLogitsProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteriaList,
)
from transformers.pipelines.text_generation import ReturnType
from transformers.utils import ModelOutput

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Response

from .._base import StreamingPipeline
from ..utils import tokenize_stopping_sequences_where_needed
from .generation import streaming_generate
from .processors import StopOnTokens
from .streaming import ResponseTokenBatchPostprocessor

logger = get_logger(__name__)


class TransformersPipeline(StreamingPipeline):
    """Stripped down version of Transformers pipeline with support for streaming.

    Args:
        model (PreTrainedModel): Hugging Face model.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        device (Optional[Union[str, int, torch.device]], optional): Device to place model on. Defaults to model's
            device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            # `accelerate` device map
            hf_device_map = getattr(self.model, "hf_device_map", None)
            if hf_device_map is not None:
                # Take the first device used by `accelerate`.
                device = next(iter(hf_device_map.values()))
            else:
                device = model.device

        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        elif device < 0:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device}")

    def _get_stopping_sequences(self, generate_kwargs: Dict[str, Any]) -> List[int]:
        if hasattr(self, "_stopping_sequences"):
            return self._stopping_sequences
        stopping_sequences = None
        if generate_kwargs.get("stopping_sequences", None) is not None:
            stopping_sequences = tokenize_stopping_sequences_where_needed(
                self.tokenizer, generate_kwargs["stopping_sequences"]
            )
        self._stopping_sequences = stopping_sequences
        return self._stopping_sequences

    def _get_stopping_criteria(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> StoppingCriteriaList:
        lst = []
        stopping_sequences = self._get_stopping_sequences(generate_kwargs)
        stopping_sequences = stopping_sequences or []
        stopping_sequences += [self.tokenizer.eos_token_id]
        lst.append(StopOnTokens(stopping_sequences))

        if generate_kwargs.get("max_time_criteria", None) is not None:
            max_time, initial_time = generate_kwargs.pop("max_time_criteria")
            lst.append(MaxTimeCriteria(max_time, initial_time))

        return StoppingCriteriaList(lst)

    def _get_logits_processors(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> LogitsProcessorList:
        lst = []
        stopping_sequences = self._get_stopping_sequences(generate_kwargs)
        if stopping_sequences and model_inputs is not None:
            min_new_tokens_stopping_sequences = []
            for sequence in stopping_sequences:
                if isinstance(sequence, list):
                    min_new_tokens_stopping_sequences.extend(sequence)
                else:
                    min_new_tokens_stopping_sequences.append(sequence)
            lst.append(
                MinNewTokensLengthLogitsProcessor(
                    prompt_length_to_skip=model_inputs["inputs"]["input_ids"].shape[1],
                    min_new_tokens=generate_kwargs.pop("min_new_tokens", 4),
                    eos_token_id=min_new_tokens_stopping_sequences
                    + [self.tokenizer.eos_token_id],
                )
            )

        return LogitsProcessorList(lst)

    def __call__(
        self,
        inputs: List[str],
        **kwargs,
    ) -> List[Response]:
        streams = [list() for _ in range(len(inputs))]
        for batch_response in self.stream(inputs, **kwargs):
            for i, response in enumerate(batch_response):
                streams[i].append(response)

        return [Response.merge_stream(*stream) for stream in streams]

    def _get_postprocessor(
        self,
        model_inputs: Dict[str, Any],
        stopping_sequences: List[int],
        **decode_kwargs,
    ) -> ResponseTokenBatchPostprocessor:
        decode_kwargs.setdefault("skip_special_tokens", True)
        decode_kwargs.setdefault("skip_prompt", True)
        return ResponseTokenBatchPostprocessor(
            tokenizer=self.tokenizer,
            preprocessing_time=model_inputs.get("preprocessing_time", None),
            stopping_sequences=stopping_sequences,
            **decode_kwargs,
        )

    @torch.inference_mode()
    def stream(
        self,
        inputs: List[str],
        **kwargs,
    ) -> Iterator[List[Response]]:
        (
            preprocess_params,
            forward_params,
            postprocess_params,
        ) = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_inputs = self._ensure_tensor_on_device(model_inputs, device=self.device)
        forward_params = self._add_default_generate_kwargs(forward_params, model_inputs)

        stopping_sequences = self._get_stopping_sequences(forward_params)
        stopping_sequences = stopping_sequences or []
        stopping_sequences += [self.tokenizer.eos_token_id]

        postprocessor = self._get_postprocessor(
            model_inputs, stopping_sequences, **postprocess_params
        )
        logger.info(
            f"Forward params: {forward_params}, batch size: {len(inputs)} model_inputs {model_inputs}"
        )
        for batch in self.forward(model_inputs, **forward_params):
            yield postprocessor.process(batch)
        yield postprocessor.end()

    def preprocess(self, prompts: List[str], **generate_kwargs):
        st = time.monotonic()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, **generate_kwargs
        ).to(self.model.device)
        if not generate_kwargs.get("return_token_type_ids", True):
            inputs.pop("token_type_ids", None)
        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "prompt_text": prompts,
            "preprocessing_time": et,
        }

    def forward(self, model_inputs, **generate_kwargs):
        inputs = model_inputs["inputs"]
        with patch(
            "transformers.generation.utils.GenerationMixin.generate", streaming_generate
        ):
            yield from self.model.generate(
                **{
                    **inputs,
                    **generate_kwargs,
                }
            )

    def ensure_tensor_on_device(self, **inputs):
        """
        Ensure PyTorch tensors are on the specified device.

        Args:
            inputs (keyword arguments that should be `torch.Tensor`, the rest is ignored):
                The tensors to place on `self.device`.
            Recursive on lists **only**.

        Return:
            `Dict[str, torch.Tensor]`: The same as `inputs` but on the proper device.
        """
        return self._ensure_tensor_on_device(inputs, self.device)

    def _ensure_tensor_on_device(self, inputs, device: torch.device):
        if isinstance(inputs, ModelOutput):
            return ModelOutput(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, dict):
            return {
                name: self._ensure_tensor_on_device(tensor, device)
                for name, tensor in inputs.items()
            }
        elif isinstance(inputs, UserDict):
            return UserDict(
                {
                    name: self._ensure_tensor_on_device(tensor, device)
                    for name, tensor in inputs.items()
                }
            )
        elif isinstance(inputs, list):
            return [self._ensure_tensor_on_device(item, device) for item in inputs]
        elif isinstance(inputs, tuple):
            return tuple(
                [self._ensure_tensor_on_device(item, device) for item in inputs]
            )
        elif isinstance(inputs, torch.Tensor):
            if device == torch.device("cpu") and inputs.dtype in {
                torch.float16,
                torch.bfloat16,
            }:
                inputs = inputs.float()
            return inputs.to(device)
        else:
            return inputs

    def _add_default_generate_kwargs(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Dict[str, Any]:
        stopping_criteria = self._get_stopping_criteria(generate_kwargs, model_inputs)
        if stopping_criteria:
            if generate_kwargs.get("stopping_criteria", None):
                generate_kwargs["stopping_criteria"].extend(stopping_criteria)
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    generate_kwargs["stopping_criteria"]
                )
            else:
                generate_kwargs["stopping_criteria"] = StoppingCriteriaList(
                    stopping_criteria
                )

        logits_processor = self._get_logits_processors(generate_kwargs, model_inputs)
        if logits_processor:
            if generate_kwargs.get("logits_processor", None):
                generate_kwargs["logits_processor"].extend(logits_processor)
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    generate_kwargs["logits_processor"]
                )
            else:
                generate_kwargs["logits_processor"] = LogitsProcessorList(
                    logits_processor
                )

        generate_kwargs.pop("stopping_sequences", None)
        return generate_kwargs

    def _sanitize_parameters(
        self,
        return_full_text=None,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        prefix=None,
        handle_long_generation=None,
        stop_sequence=None,
        # New aviary arguments
        return_token_type_ids=None,
        stopping_sequences=None,
        add_special_tokens=None,
        timeout_s=None,
        start_timestamp=None,
        **generate_kwargs,
    ):
        preprocess_params = {}
        if prefix is not None:
            preprocess_params["prefix"] = prefix
        if return_token_type_ids is not None:
            preprocess_params["return_token_type_ids"] = return_token_type_ids
        if add_special_tokens is not None:
            preprocess_params["add_special_tokens"] = add_special_tokens
        if prefix:
            prefix_inputs = self.tokenizer(
                prefix, padding=False, add_special_tokens=False, return_tensors="pt"
            )
            prefix_length = prefix_inputs["input_ids"].shape[-1]

            if "max_new_tokens" in generate_kwargs:
                pass
            elif "max_length" in generate_kwargs:
                generate_kwargs["max_length"] += prefix_length
            else:
                generate_kwargs["max_length"] = (
                    self.model.config.max_length + prefix_length
                )

            if "min_length" in generate_kwargs:
                generate_kwargs["min_length"] += prefix_length
        if handle_long_generation is not None:
            if handle_long_generation not in {"hole"}:
                raise ValueError(
                    f"{handle_long_generation} is not a valid value for `handle_long_generation` parameter expected"
                    " [None, 'hole']"
                )
            preprocess_params["handle_long_generation"] = handle_long_generation

        if stopping_sequences is not None:
            generate_kwargs["stopping_sequences"] = stopping_sequences

        if timeout_s is not None and start_timestamp is not None:
            generate_kwargs["max_time_criteria"] = (timeout_s, start_timestamp)

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_full_text is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_full_text`"
                )
            if return_tensors is not None:
                raise ValueError(
                    "`return_full_text` is mutually exclusive with `return_tensors`"
                )
            return_type = (
                ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
            )
        if return_tensors is not None and return_type is None:
            if return_text is not None:
                raise ValueError(
                    "`return_text` is mutually exclusive with `return_tensors`"
                )
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if clean_up_tokenization_spaces is not None:
            postprocess_params[
                "clean_up_tokenization_spaces"
            ] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(
                stop_sequence, add_special_tokens=False
            )
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim.",
                    stacklevel=2,
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params
