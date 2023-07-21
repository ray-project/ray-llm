from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from aviary.backend.llm.continuous.error_handling import ErrorReason
from aviary.backend.llm.continuous.scheduler import Request
from aviary.backend.logger import get_logger
from aviary.backend.server.models import Response

from ._base import AsyncStreamingPipeline
from .utils import (
    decode_stopping_sequences_where_needed,
    pythonize_tensors,
)

try:
    from text_generation_server.pb.generate_pb2 import (
        NextTokenChooserParameters,
        StoppingCriteriaParameters,
    )
    from text_generation_server.pb.generate_pb2 import (
        Request as GenerationRequest,
    )
except ImportError as e:
    GenerationRequest = e
    NextTokenChooserParameters = None
    StoppingCriteriaParameters = None

logger = get_logger(__name__)

if TYPE_CHECKING:
    from text_generation_server.models.types import (
        Generation,
    )

    from aviary.backend.llm.continuous.tgi.tgi_worker import TGIInferenceWorker


@dataclass
class AviaryGenerationRequestWrapper:
    """Wrapper for GenerationRequest that extra Aviary fields."""

    gen_request: GenerationRequest
    min_new_tokens: int = 8

    def __getattr__(self, name):
        return getattr(self.gen_request, name)


class TextGenerationInferencePipeline(AsyncStreamingPipeline):
    """Text generation pipeline using Continuous Batching."""

    def __init__(
        self,
        model: "TGIInferenceWorker",
        tokenizer,
        device: Union[str, int, torch.device, None] = None,
    ) -> None:
        if isinstance(GenerationRequest, Exception):
            raise RuntimeError(
                "TextGenerationInferencePipeline requires text-generation-inference to be installed."
            ) from GenerationRequest

        # TODO don't use private APIs here
        tokenizer = model._model.tokenizer

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        super().__init__(model, tokenizer, device)

    def get_input_length(self, input_text: str) -> int:
        return self.tokenizer(
            text=input_text,
            return_tensors="np",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
        )["input_ids"].shape[1]

    def _parse_sampling_args(
        self, generate_kwargs: Dict[str, Any], model_inputs=None
    ) -> Tuple[
        "NextTokenChooserParameters", "StoppingCriteriaParameters", Dict[str, Any]
    ]:
        temperature = generate_kwargs.pop("temperature", 1.0)
        repetition_penalty = generate_kwargs.pop("repetition_penalty", 1.1)
        top_k = generate_kwargs.pop("top_k", 0)
        top_p = generate_kwargs.pop("top_p", 1.0)
        typical_p = generate_kwargs.pop("typical_p", 1.0)
        do_sample = generate_kwargs.pop("do_sample", False)
        max_new_tokens = generate_kwargs.pop("max_new_tokens", 256)
        stop_sequences = generate_kwargs.pop("stopping_sequences", [])
        ignore_eos_token = generate_kwargs.pop("ignore_eos_token", False)
        watermark = generate_kwargs.pop("watermark", False)
        seed = generate_kwargs.pop("seed", 42)

        stop_sequences = decode_stopping_sequences_where_needed(
            self.tokenizer, stop_sequences
        )
        if not do_sample:
            temperature = 1.0
        elif temperature <= 0:
            temperature = 0.01

        parameters = NextTokenChooserParameters(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            do_sample=do_sample,
            seed=seed,
            repetition_penalty=repetition_penalty,
            watermark=watermark,
        )
        stopping_parameters = StoppingCriteriaParameters(
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences,
            ignore_eos_token=ignore_eos_token,
        )

        return parameters, stopping_parameters, generate_kwargs

    # TODO Fix this
    def __call__(self, inputs: List[str], **kwargs) -> List[Response]:
        raise NotImplementedError

    def _parse_requests(
        self, requests: List["Request"], *, verbose: bool = True
    ) -> List["GenerationRequest"]:
        parsed_requests = []
        for r in requests:
            generate_params = r.params.copy()
            generate_params["max_new_tokens"] = r.max_new_tokens
            generate_params["seed"] = generate_params.get("seed", r.id)
            (
                parameters,
                stopping_parameters,
                generate_params,
            ) = self._parse_sampling_args(generate_params)
            if verbose:
                logger.info(
                    f"id: {r.id} parameters: {parameters}, stopping_parameters: {stopping_parameters} model_inputs {r.inputs}"
                )
            parsed_request = AviaryGenerationRequestWrapper(
                GenerationRequest(
                    id=r.id,
                    inputs=r.inputs,
                    truncate=r.truncate,
                    prefill_logprobs=True,
                    parameters=parameters,
                    stopping_parameters=stopping_parameters,
                ),
                min_new_tokens=generate_params.get("min_new_tokens", 16),
            )
            parsed_requests.append(parsed_request)
        return parsed_requests

    def process_new_batch(
        self, requests: List["Request"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        parsed_requests = self._parse_requests(requests)
        generations, id = self.model.process_new_batch(parsed_requests, batch_id)
        if generations and isinstance(generations, list):
            generations = [pythonize_tensors(g) for g in generations]
        return generations, id

    def requires_padding(self) -> bool:
        return self.model.requires_padding()

    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        generations, id = self.model.generate_next_token(batch_ids)
        if generations and isinstance(generations, list):
            generations = [pythonize_tensors(g) for g in generations]
        return generations, id

    def filter_requests(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        return self.model.filter_requests(batch_id, request_ids)

    def warmup(
        self, requests: List["Request"], batch_id: int, max_total_tokens: int
    ) -> Optional[int]:
        parsed_requests = self._parse_requests(requests, verbose=False)
        return self.model.warmup(parsed_requests, batch_id, max_total_tokens)
