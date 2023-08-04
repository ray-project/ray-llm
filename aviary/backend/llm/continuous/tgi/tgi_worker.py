import asyncio
import gc
import os
import traceback
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import ray
import torch
from filelock import FileLock
from text_generation_server.pb.generate_pb2 import (
    Request as TGIRequest,
)

from aviary.backend.llm.utils import (
    decode_stopping_sequences_where_needed,
    pythonize_tensors,
)
from aviary.backend.logger import get_logger

from ..error_handling import ErrorReason, OutOfMemory
from ..types import InferenceTask, Request, TGIParams
from ..worker import AbstractInferenceWorker, AsyncInferenceWorker
from .logits_processors import (
    HeterogeneousFrequencyPresencePenaltyLogitsProcessor,
    MinNewTokensLogitsProcessor,
)

if TYPE_CHECKING:
    from text_generation_server.models.causal_lm import CausalLMBatch
    from text_generation_server.models.model import Model
    from text_generation_server.models.types import (
        Generation,
    )
    from text_generation_server.pb.generate_pb2 import (
        NextTokenChooserParameters,
        StoppingCriteriaParameters,
    )

logger = get_logger(__name__)


@dataclass
class FakePB2:
    requests: List[TGIRequest]
    id: Optional[int] = None
    request_ids: Optional[List[int]] = None
    max_tokens: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.requests) if self.requests else 0


@dataclass
class AviaryGenerationRequestWrapper:
    """Wrapper for TGIRequest that extra Aviary fields."""

    gen_request: TGIRequest
    min_new_tokens: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __getattr__(self, name):
        return getattr(self.gen_request, name)


@dataclass
class PagedAttentionInferenceTask(InferenceTask):
    # same as text_generation_server.models.flash_causal_lm.BLOCK_SIZE
    # Ideally we would just import that, but it will cause CUDA to be loaded
    # on the scheduler actor, which is not ideal.
    block_size: int = 16

    @property
    def decode_cost(self) -> int:
        return int(
            ((self.request.max_new_tokens + self.block_size - 1) / self.block_size)
            * self.block_size
        )

    @property
    def input_cost(self) -> int:
        return int(
            ((self.input_length + self.block_size - 1) / self.block_size)
            * self.block_size
        )


def postprocess_batch(batch: "CausalLMBatch") -> "CausalLMBatch":
    """
    Postprocess batch object.

    Currently, it injects the MinNewTokens processor to prevent models like
    vicuna returning empty strings.
    """
    from text_generation_server.utils.logits_process import (
        HeterogeneousProcessorWrapper,
    )
    from text_generation_server.utils.tokens import (
        HeterogeneousNextTokenChooser,
        NextTokenChooser,
    )

    class MinLenNextTokenChooser(NextTokenChooser):
        def __call__(self, input_ids, scores):
            scores = self.min_new_length_preprocessor(input_ids, scores)
            return super().__call__(input_ids, scores)

    # TODO support MinNewTokens for stopping sequences in addition to eos_token_id
    if hasattr(batch, "next_token_chooser") and isinstance(
        batch.next_token_chooser, HeterogeneousNextTokenChooser
    ):
        # This is hacky
        batch.next_token_chooser.warpers.append(
            HeterogeneousProcessorWrapper(
                {
                    i: MinNewTokensLogitsProcessor(
                        batch.stopping_criterias[i].current_tokens,
                        batch.requests[i].min_new_tokens,
                        batch.stopping_criterias[i].eos_token_id,
                    )
                    for i in range(len(batch.input_lengths))
                }
            )
        )
        presence_penalty = [r.presence_penalty for r in batch.requests]
        frequency_penalty = [r.frequency_penalty for r in batch.requests]
        if any([x != 0.0 for x in presence_penalty]) or any(
            [x != 0.0 for x in frequency_penalty]
        ):
            batch.next_token_chooser.repetition_processor = (
                HeterogeneousFrequencyPresencePenaltyLogitsProcessor(
                    presence_penalty,
                    frequency_penalty,
                    dtype=batch.next_token_chooser.dtype,
                    device=batch.next_token_chooser.device,
                )
            )
    elif hasattr(batch, "next_token_choosers") and isinstance(
        batch.next_token_choosers[0], NextTokenChooser
    ):
        for i in range(len(batch.next_token_choosers)):
            # This is super hacky
            batch.next_token_choosers[i].__class__ = MinLenNextTokenChooser
            batch.next_token_choosers[
                i
            ].min_new_length_preprocessor = MinNewTokensLogitsProcessor(
                batch.stopping_criterias[i].current_tokens,
                batch.requests[i].min_new_tokens,
                batch.stopping_criterias[i].eos_token_id,
            )
        # TODO support frequency and presence penalty for non-flash models
    return batch


def _format_batch(batch: "CausalLMBatch") -> str:
    return f"{batch.batch_id} {batch.requests} {batch.requests_idx_mapping}"


def create_batch(
    model: "Model", requests: List["TGIRequest"], batch_id: int
) -> Type["CausalLMBatch"]:
    batch = model.batch_type.from_pb(
        FakePB2(id=batch_id, requests=requests),
        tokenizer=model.tokenizer,
        dtype=model.dtype,
        device=model.device,
    )
    logger.debug(f"Created batch {_format_batch(batch)}")
    batch = postprocess_batch(batch)
    return batch


def concatenate_batches(
    model: "Model", batches: List["CausalLMBatch"]
) -> "CausalLMBatch":
    # Reset batch_id
    logger.debug(
        f"Concatenating {[_format_batch(b) for b in batches]} into {batches[0].batch_id}"
    )
    new_batch = model.batch_type.concatenate(batches)
    logger.debug(f"Concatenated batch {_format_batch(new_batch)}")
    new_batch = postprocess_batch(new_batch)
    return new_batch


def _flatten_list(lst: list) -> list:
    return [item for sublist in lst for item in sublist]


class TGIInferenceWorkerGroup(AsyncInferenceWorker):
    def __init__(self, worker_group: List[ray.ObjectRef]):
        self.worker_group = worker_group
        self._requires_padding = ray.get(self.worker_group[0].requires_padding.remote())
        self._can_infer_max_batch_total_tokens = ray.get(
            self.worker_group[0].can_infer_max_batch_total_tokens.remote()
        )

    def can_infer_max_batch_total_tokens(self):
        return self._can_infer_max_batch_total_tokens

    def requires_padding(self) -> bool:
        return self._requires_padding

    def get_inference_task_cls(self) -> Type[InferenceTask]:
        return ray.get(self.worker_group[0].get_inference_task_cls.remote())

    def _parse_worker_group_return(self, ret):
        potential_error = next(
            (x[0] for x in ret if isinstance(x[0], ErrorReason)), None
        )
        if potential_error:
            return potential_error, None
        return (
            _flatten_list([x[0] for x in ret])
            if isinstance(ret[0][0], list)
            else ret[0][0],
            ret[0][1],
        )

    def process_new_batch(
        self, requests: List["Request"], batch_id: int
    ) -> Tuple[Union[List["Generation"], ErrorReason], int]:
        ret = ray.get(
            [
                worker.process_new_batch.remote(requests, batch_id)
                for worker in self.worker_group
            ]
        )
        return self._parse_worker_group_return(ret)

    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        ret = ray.get(
            [
                worker.generate_next_token.remote(batch_ids)
                for worker in self.worker_group
            ]
        )
        return self._parse_worker_group_return(ret)

    def filter_tasks(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        return ray.get(
            [
                worker.filter_tasks.remote(batch_id, request_ids)
                for worker in self.worker_group
            ]
        )[0]

    async def process_new_batch_async(
        self, requests: List["Request"], batch_id: int
    ) -> Tuple[Union[List["Generation"], ErrorReason], int]:
        done, pending = await asyncio.wait(
            [
                worker.process_new_batch.remote(requests, batch_id)
                for worker in self.worker_group
            ],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        done = await asyncio.gather(*done)
        return self._parse_worker_group_return(done)

    async def generate_next_token_async(
        self, batch_ids: List[int]
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        done, pending = await asyncio.wait(
            [
                worker.generate_next_token.remote(batch_ids)
                for worker in self.worker_group
            ],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        done = await asyncio.gather(*done)
        return self._parse_worker_group_return(done)

    async def filter_tasks_async(
        self, batch_id: int, request_ids: List[int]
    ) -> Optional[int]:
        done, pending = await asyncio.wait(
            [
                worker.filter_tasks.remote(batch_id, request_ids)
                for worker in self.worker_group
            ],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        return (await asyncio.gather(*done))[0]


class TGIInferenceWorker(AbstractInferenceWorker):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        sharded: Optional[bool] = None,
        quantize: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        self.model_id = model_id
        self.revision = revision
        self.sharded = sharded
        self.quantize = quantize
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        # We need to do lazy init in init_model to ensure that model
        # is loaded after CUDA_VISIBLE_DEVICES is set by Ray.

    def _set_cuda_device(function: Callable):
        @wraps(function)
        def wrapper(self: "TGIInferenceWorker", *args, **kwargs):
            if self.current_device:
                torch.cuda.set_device(self.current_device)
            return function(self, *args, **kwargs)

        return wrapper

    def _pythonize_outputs(function: Callable):
        @wraps(function)
        def wrapper(self: "TGIInferenceWorker", *args, **kwargs):
            ret = function(self, *args, **kwargs)
            return pythonize_tensors(ret)

        return wrapper

    def init_model(
        self,
        local_rank: int,
        num_cpus_per_worker: int = 1,
        num_gpus_per_worker: float = 0,
    ):
        os.environ["OMP_NUM_THREADS"] = str(int(num_cpus_per_worker))
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        expected_device = None
        if torch.cuda.is_available() and num_gpus_per_worker > 0:
            self.gpu_memory_fraction = float(min(num_gpus_per_worker, 1.0))
            os.environ["CUDA_MEMORY_FRACTION"] = str(self.gpu_memory_fraction)
            expected_device = torch.device(f"cuda:{local_rank}")
        self.local_rank = local_rank

        logger.info(f"Loading model from {self.model_id}")

        from text_generation_server.cli import download_weights
        from text_generation_server.models import get_model

        lock_path = os.path.expanduser(f"~/{self.model_id.replace('/', '--')}.lock")
        with FileLock(lock_path):
            download_weights(
                model_id=self.model_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
            )

        self._model = get_model(
            model_id=self.model_id,
            revision=self.revision,
            sharded=int(os.getenv("WORLD_SIZE", "1")) > 1
            if self.sharded is None
            else self.sharded,
            quantize=self.quantize,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )

        self._batch_state_cache: Dict[int, "CausalLMBatch"] = dict()
        if self._model.device.type == "cuda":
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

        if torch.cuda.is_available():
            # Save the current device so we can set it again if it gets reset
            self.current_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            assert expected_device is None or self.current_device == expected_device

    def requires_padding(self) -> bool:
        return self._model.requires_padding

    def get_inference_task_cls(self) -> Type[InferenceTask]:
        if not self._model.requires_padding:
            return PagedAttentionInferenceTask
        return InferenceTask

    def _handle_model_exception(
        self, e: Exception, clear_batch_state: bool = False
    ) -> Tuple[ErrorReason, None]:
        if (
            isinstance(e, torch.cuda.OutOfMemoryError)
            or "CUDA" in str(e)
            or "cache blocks" in str(e)
        ):
            logger.error(f"OOM error happened: {traceback.format_exc()}")
            if clear_batch_state:
                self._batch_state_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return OutOfMemory(str(e)), None
        else:
            raise e

    def _parse_sampling_params(
        self, params: TGIParams, max_new_tokens: int
    ) -> Tuple[
        "NextTokenChooserParameters", "StoppingCriteriaParameters", Dict[str, Any]
    ]:
        from text_generation_server.pb.generate_pb2 import (
            NextTokenChooserParameters,
            StoppingCriteriaParameters,
        )

        temperature = params.temperature
        top_k = params.top_k
        top_p = params.top_p
        typical_p = params.typical_p
        do_sample = params.do_sample
        repetition_penalty = params.repetition_penalty
        stop_sequences = params.stop_sequences
        ignore_eos_token = params.ignore_eos_token
        watermark = params.watermark
        seed = params.seed

        stop_sequences = decode_stopping_sequences_where_needed(
            self._model.tokenizer, stop_sequences
        )

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

        return parameters, stopping_parameters

    def _parse_requests(
        self, requests: List["Request"], *, verbose: bool = True
    ) -> List["TGIRequest"]:
        parsed_requests = []
        for r in requests:
            if r.params.seed is None:
                r.params.seed = r.id
            (
                parameters,
                stopping_parameters,
            ) = self._parse_sampling_params(r.params, r.max_new_tokens)
            if verbose:
                logger.info(
                    f"id: {r.id} parameters: {parameters}, stopping_parameters: {stopping_parameters} model_inputs {r.inputs}"
                )
            parsed_request = AviaryGenerationRequestWrapper(
                TGIRequest(
                    id=r.id,
                    inputs=r.inputs,
                    truncate=r.truncate,
                    prefill_logprobs=True,
                    parameters=parameters,
                    stopping_parameters=stopping_parameters,
                ),
                min_new_tokens=r.params.min_new_tokens,
                frequency_penalty=r.params.frequency_penalty,
                presence_penalty=r.params.presence_penalty,
            )
            parsed_requests.append(parsed_request)
        return parsed_requests

    @_set_cuda_device
    @_pythonize_outputs
    def process_new_batch(
        self, requests: List["TGIRequest"], batch_id: int
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        requests = self._parse_requests(requests, verbose=False)
        # TGI expects sorted requests
        requests = sorted(requests, key=lambda x: x.id)
        logger.info(f"Processing new batch {batch_id}:\n{requests}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_state = create_batch(self._model, requests, batch_id)
        # Prefill
        try:
            generations, batch_state = self._model.generate_token(batch_state)
        except Exception as e:
            return self._handle_model_exception(e)

        if batch_state is not None:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        else:
            return generations, None

    async def process_new_batch_async(
        self, requests: List["TGIRequest"], batch_id: int
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        return self.process_new_batch(requests, batch_id)

    @_set_cuda_device
    @_pythonize_outputs
    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        if len(batch_ids) == 0:
            raise ValueError("Must provide at least one batch")
        batch_states = []
        for batch_id in batch_ids:
            if batch_id is None:
                continue
            batch_state = self._batch_state_cache.pop(batch_id, None)
            if batch_state is None:
                raise ValueError(f"Batch ID {batch_id} not found in cache.")
            batch_states.append(batch_state)

        if len(batch_states) == 0:
            return [], None

        try:
            if len(batch_states) > 1:
                batch_state = concatenate_batches(self._model, batch_states)
            else:
                batch_state = batch_states[0]
            generations, batch_state = self._model.generate_token(batch_state)
        except Exception as e:
            return self._handle_model_exception(e, clear_batch_state=True)

        if batch_state:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        return generations, None

    async def generate_next_token_async(
        self, batch_ids: List[int]
    ) -> Tuple[Union[List["Generation"], ErrorReason], Optional[int]]:
        return self.generate_next_token(batch_ids)

    @_set_cuda_device
    def filter_tasks(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        """Update the batch state cache to only contain the unfinished tasks."""
        if batch_id is None:
            return None

        batch_state = self._batch_state_cache.pop(batch_id)

        if len(request_ids) == 0:
            return None

        # TGI expects sorted request_ids
        request_ids = sorted(request_ids)
        # only these request ids will remain in the batch_state
        # these request ids are the belong to unfinished tasks
        filtered = batch_state.filter(request_ids)
        logger.debug(
            f"Filtered batch {_format_batch(batch_state)} into {_format_batch(filtered)}"
        )

        if len(filtered):
            self._batch_state_cache[filtered.batch_id] = filtered
            return filtered.batch_id

        return None

    async def filter_tasks_async(
        self, batch_id: int, request_ids: List[int]
    ) -> Optional[int]:
        return self.filter_tasks(batch_id, request_ids)

    @_set_cuda_device
    def warmup(
        self,
        max_batch_prefill_tokens: int,
        max_input_length: int,
        max_batch_total_tokens: Optional[int] = None,
    ) -> int:
        n_tokens = 0
        requests = []
        while n_tokens < max_batch_prefill_tokens:
            num_input_tokens = min(
                max_input_length,
                max_batch_prefill_tokens - n_tokens,
            )
            requests.append(
                Request(
                    id=0,
                    inputs="_test " * num_input_tokens,
                    truncate=num_input_tokens,
                    params=TGIParams(
                        temperature=0.9,
                        top_k=10,
                        top_p=0.9,
                        typical_p=0.9,
                        do_sample=False,
                        seed=0,
                        repetition_penalty=1.2,
                        watermark=True,
                        stop_sequences=[],
                        ignore_eos_token=False,
                    ),
                    max_new_tokens=2,
                    input_tokens=num_input_tokens,
                )
            )
            n_tokens += num_input_tokens

        logger.info(
            f"Model is warming up. Num requests: {len(requests)} Prefill tokens: {n_tokens} Max batch total tokens: {max_batch_total_tokens}"
        )

        requests = sorted(
            self._parse_requests(requests, verbose=False), key=lambda x: x.id
        )
        batch_state = create_batch(self._model, requests, 0)
        suggested_max_batch_total_tokens = self._model.warmup(
            batch_state, max_batch_total_tokens
        )
        if not suggested_max_batch_total_tokens:
            suggested_max_batch_total_tokens = max_batch_total_tokens
        logger.info(
            f"Model finished warming up (max_batch_total_tokens={suggested_max_batch_total_tokens}) and is ready to serve requests."
        )
        return suggested_max_batch_total_tokens

    def check_cuda_objects(self):
        from collections import defaultdict

        if self._model.device.type != "cuda":
            return
        d = defaultdict(int)

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    t = tuple(obj.size()) + (obj.dtype, obj.device)
                    d[t] += 1
            except Exception:
                pass

        for count, obj_signature in sorted(
            [(count, sig) for sig, count in d.items()], key=lambda x: x[0], reverse=True
        ):
            logger.info(count, obj_signature)

    def debug_objects(self):
        objs = gc.get_objects()
        tensors = [obj for obj in objs if torch.is_tensor(obj)]
        leaked_tensors = [t for t in tensors if t.size() == torch.Size([20, 1, 1024])]
        if len(leaked_tensors) >= 1000:
            import pdb

            pdb.set_trace()

    def report_stats(self):
        if self._model.device.type == "cuda":
            # gc.collect()
            logger.info(
                f"memory allocated: {torch.cuda.memory_allocated(self._model.device) / 2 ** 30}"
            )
            logger.info(
                f"memory reserved: {torch.cuda.memory_reserved(self._model.device) / 2 ** 30}"
            )
            # self.check_cuda_objects()
            # if torch.cuda.memory_allocated(self._model.device) / 2 ** 30 > 30:
            #    self.debug_objects()

    def get_tokenizer(self):
        return self._model.tokenizer

    def can_infer_max_batch_total_tokens(self):
        return not self._model.requires_padding
