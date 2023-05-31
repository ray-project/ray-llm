import time
from typing import List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aviary.backend.logger import get_logger
from aviary.backend.server.models import Response

from ._base import BasePipeline
from .processors import StopOnTokens
from .utils import construct_prompts, truncate_to_first_stop_token

logger = get_logger(__name__)


class DefaultPipeline(BasePipeline):
    """Default text generation pipeline.

    Args:
        model (PreTrainedModel): Hugging Face model.
        tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
        prompt_format (Optional[str], optional): Prompt format. Defaults to None.
        device (Optional[Union[str, int, torch.device]], optional): Device to place model on. Defaults to model's
            device.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_format: Optional[str] = None,
        device: Optional[Union[str, int, torch.device]] = None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            device=device,
        )

    def preprocess(self, prompts: List[str], **generate_kwargs):
        st = time.monotonic()
        prompt_text = construct_prompts(prompts, prompt_format=self.prompt_format)
        instruction_text = construct_prompts(prompts, prompt_format="")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", padding=True, **generate_kwargs
        ).to(self.model.device)
        if not generate_kwargs.get("return_token_type_ids", True):
            inputs.pop("token_type_ids", None)
        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": et,
        }

    def forward(self, model_inputs, **generate_kwargs):
        st = time.monotonic()
        inputs = model_inputs["inputs"]
        instruction_text = model_inputs["instruction_text"]
        prompt_text = model_inputs["prompt_text"]
        preprocessing_time = model_inputs["preprocessing_time"]

        generated_sequence = self.model.generate(
            **{
                **inputs,
                **generate_kwargs,
            }
        )
        et = time.monotonic() - st
        return {
            "inputs": inputs,
            "generated_sequence": generated_sequence,
            "instruction_text": instruction_text,
            "prompt_text": prompt_text,
            "preprocessing_time": preprocessing_time,
            "generation_time": et,
            "generate_kwargs": generate_kwargs,
        }

    def postprocess(self, model_outputs, **postprocess_kwargs) -> List[Response]:
        st = time.monotonic()
        tokens = model_outputs["generated_sequence"]
        input_ids = model_outputs["inputs"]["input_ids"]
        token_stopper = next(
            (
                x
                for x in model_outputs["generate_kwargs"]["stopping_criteria"]
                if isinstance(x, StopOnTokens)
            ),
            None,
        )
        decoded: List[Response] = []
        num_generated_tokens_batch = 0
        num_input_tokens_batch = 0
        for token_unwrapped, inputs_unwrapped in zip(tokens, input_ids):
            logger.info(
                f"Unprocessed generated tokens: '{self.tokenizer.decode(token_unwrapped, skip_special_tokens=False).encode('unicode_escape').decode('utf-8')}'"
            )
            tokens = token_unwrapped[len(inputs_unwrapped) :]
            if token_stopper:
                tokens = truncate_to_first_stop_token(
                    tokens, token_stopper.stopping_sequences
                )
            text = (
                self.tokenizer.decode(tokens, skip_special_tokens=True)
                .replace("\u200b", "")
                .strip()
            )
            for i in range(len(inputs_unwrapped)):
                if inputs_unwrapped[i] != self.tokenizer.pad_token_id:
                    break
            num_input_tokens = len(inputs_unwrapped[i:])
            num_generated_tokens = len(tokens)
            response = Response(
                generated_text=text,
                num_generated_tokens=num_generated_tokens,
                num_input_tokens=num_input_tokens,
            )
            num_generated_tokens_batch += num_generated_tokens
            num_input_tokens_batch += num_input_tokens
            decoded.append(response)
        et = time.monotonic() - st
        for response in decoded:
            response.num_generated_tokens_batch = num_generated_tokens_batch
            response.num_input_tokens_batch = num_input_tokens_batch
            response.preprocessing_time = model_outputs["preprocessing_time"]
            response.generation_time = model_outputs["generation_time"]
            response.postprocessing_time = et
        return decoded
