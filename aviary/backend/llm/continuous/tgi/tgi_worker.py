import asyncio
import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import ray
import torch
from filelock import FileLock

from aviary.backend.logger import get_logger

from ..types import Request
from ..worker import AsyncInferenceWorker

if TYPE_CHECKING:
    from text_generation_server.models.causal_lm import CausalLMBatch
    from text_generation_server.models.model import Model
    from text_generation_server.models.types import (
        Generation,
    )


from text_generation_server.pb.generate_pb2 import (
    Request as GenerationRequest,
)

from ..error_handling import ErrorReason, OutOfMemory
from ..worker import AbstractInferenceWorker
from .logits_processors import MinNewTokensLogitsProcessor

logger = get_logger(__name__)


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
    return batch


def _print_batch(batch):
    return f"{batch.batch_id} {batch.requests} {batch.requests_idx_mapping}"


def create_batch(
    model: "Model", requests: List["GenerationRequest"], batch_id: int
) -> Type["CausalLMBatch"]:
    batch = model.batch_type.from_pb(
        FakePB2(id=batch_id, requests=requests),
        tokenizer=model.tokenizer,
        dtype=model.dtype,
        device=model.device,
    )
    logger.debug(f"Created batch {_print_batch(batch)}")
    batch = postprocess_batch(batch)
    return batch


def concatenate_batches(
    model: "Model", batches: List["CausalLMBatch"]
) -> "CausalLMBatch":
    # Reset batch_id
    logger.debug(
        f"Concatenating {[_print_batch(b) for b in batches]} into {batches[0].batch_id}"
    )
    new_batch = model.batch_type.concatenate(batches)
    logger.debug(f"Concatenated batch {_print_batch(new_batch)}")
    new_batch = postprocess_batch(new_batch)
    return new_batch


@dataclass
class FakePB2:
    requests: List[GenerationRequest]
    id: Optional[int] = None
    request_ids: Optional[List[int]] = None
    max_tokens: Optional[int] = None

    @property
    def size(self) -> int:
        return len(self.requests) if self.requests else 0


def _flatten_list(lst: list) -> list:
    return [item for sublist in lst for item in sublist]


# TODO: Add error handling.
# We need to catch the exception and propagate it to the user.
class TGIRayInferenceWorker(AsyncInferenceWorker):
    def __init__(self, worker_group: List[ray.ObjectRef]):
        self.worker_group = worker_group
        self._requires_padding = ray.get(self.worker_group[0].requires_padding.remote())

    def requires_padding(self) -> bool:
        return self._requires_padding

    def process_new_batch(
        self, requests: List["Request"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        ret = ray.get(
            [
                worker.process_new_batch.remote(requests, batch_id)
                for worker in self.worker_group
            ]
        )
        return (
            _flatten_list([x[0] for x in ret])
            if isinstance(ret[0][0], list)
            else ret[0][0],
            ret[0][1],
        )

    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        ret = ray.get(
            [
                worker.generate_next_token.remote(batch_ids)
                for worker in self.worker_group
            ]
        )
        return (
            _flatten_list([x[0] for x in ret])
            if isinstance(ret[0][0], list)
            else ret[0][0],
            ret[0][1],
        )

    def filter_requests(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        return ray.get(
            [
                worker.filter_requests.remote(batch_id, request_ids)
                for worker in self.worker_group
            ]
        )[0]

    async def process_new_batch_async(
        self, requests: List["Request"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], int]:
        ret = await asyncio.gather(
            *[
                worker.process_new_batch.remote(requests, batch_id)
                for worker in self.worker_group
            ]
        )
        logger.debug(f"process_new_batch_async returns {ret}")
        return (
            _flatten_list([x[0] for x in ret])
            if isinstance(ret[0][0], list)
            else ret[0][0],
            ret[0][1],
        )

    async def generate_next_token_async(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        ret = await asyncio.gather(
            *[
                worker.generate_next_token.remote(batch_ids)
                for worker in self.worker_group
            ]
        )
        logger.debug(f"generate_next_token_async returns {ret}")
        return (
            _flatten_list([x[0] for x in ret])
            if isinstance(ret[0][0], list)
            else ret[0][0],
            ret[0][1],
        )

    async def filter_requests_async(
        self, batch_id: int, request_ids: List[int]
    ) -> Optional[int]:
        ret = await asyncio.gather(
            *[
                worker.filter_requests.remote(batch_id, request_ids)
                for worker in self.worker_group
            ]
        )
        logger.debug(f"filter_requests_async returns {ret}")
        return ret[0]


class InferenceWorker(AbstractInferenceWorker):
    def __init__(self, model_loader: Callable[[], "Model"]):
        self._model = model_loader()
        self._batch_state_cache: Dict[int, "CausalLMBatch"] = dict()
        if self._model.device.type == "cuda":
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    def process_new_batch(
        self, requests: List["GenerationRequest"], batch_id: int
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
        # TGI expects sorted requests
        requests = sorted(requests, key=lambda x: x.id)
        batch_state = create_batch(self._model, requests, batch_id)
        try:
            generations, batch_state = self._model.generate_token(batch_state)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM error happened: {repr(e)}")
            return OutOfMemory(str(e)), None
        except Exception as e:
            if "CUDA" in str(e) or "cache blocks" in str(e):
                logger.error(f"CUDA error happened, treating as OOM error: {repr(e)}")
                return OutOfMemory(str(e)), None
            # logger.error(f"generate_next_token error happened: {repr(e)}")
            # return IrrecoverableError(str(e)), None
            raise
        try:
            logger.debug(f"Batch state ID: {batch_state.batch_id}")
        except Exception as e:
            logger.error(f"Failed to get batch state id {repr(e)}")
        logger.debug(
            f"process_new_batch returns {(generations, batch_state.batch_id if batch_state else None)}"
        )
        if batch_state is not None:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        else:
            return generations, None

    def generate_next_token(
        self, batch_ids: List[int]
    ) -> Tuple[List[Union["Generation", ErrorReason]], Optional[int]]:
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
            # stats = batch_state.stats()
            # logger.info(f"generate_next_token batch_state { batch_state}")
            generations, batch_state = self._model.generate_token(batch_state)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM error happened: {repr(e)}")
            self._batch_state_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return OutOfMemory(str(e)), None
        except Exception as e:
            self._batch_state_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if "CUDA" in str(e) or "cache blocks" in str(e):
                logger.error(f"CUDA error happened, treating as OOM error: {repr(e)}")
                return OutOfMemory(str(e)), None
            # logger.error(f"generate_next_token error happened: {repr(e)}")
            # return IrrecoverableError(str(e)), None
            raise

        logger.debug(
            f"generate_next_token returns {(generations, batch_state.batch_id if batch_state else None)}"
        )

        if batch_state:
            self._batch_state_cache[batch_state.batch_id] = batch_state
            return generations, batch_state.batch_id
        return generations, None

    def filter_requests(self, batch_id: int, request_ids: List[int]) -> Optional[int]:
        if batch_id is None:
            return None

        batch_state = self._batch_state_cache.pop(batch_id)

        if len(request_ids) == 0:
            return None

        # TGI expects sorted request_ids
        request_ids = sorted(request_ids)
        filtered = batch_state.filter(request_ids)
        logger.debug(
            f"Filtered batch {_print_batch(batch_state)} into {_print_batch(filtered)}"
        )
        if len(filtered):
            self._batch_state_cache[filtered.batch_id] = filtered
            return filtered.batch_id

        return None

    def warmup(
        self, requests: List["GenerationRequest"], batch_id: int, max_total_tokens: int
    ) -> Optional[int]:
        """
        Warmup is NOT optional, as it initializes the vllm cache with the correct size.
        """
        # TGI expects sorted requests
        requests = sorted(requests, key=lambda x: x.id)
        batch_state = create_batch(self._model, requests, batch_id)
        self._model.warmup(batch_state, max_total_tokens)
        return True

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
        # print(f"worker stats: {[(id, cache.stats()) for id, cache in self._batch_state_cache.items()]}")
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


class TGIInferenceWorker(InferenceWorker):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        sharded: Optional[bool] = None,
        quantize: Optional[str] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        from text_generation_server.cli import download_weights
        from text_generation_server.models import get_model

        lock_path = os.path.expanduser(f"~/{model_id.replace('/', '--')}.lock")
        with FileLock(lock_path):
            download_weights(model_id=model_id, revision=revision)

        from huggingface_hub import hf_hub_download as hf_hub_download_original
        from transformers import AutoModelForCausalLM

        # Force device_map="auto" even for single GPU models
        # to increase loading speed drastically
        from_pretrained_original = AutoModelForCausalLM.from_pretrained

        def from_pretrained(*args, **kwargs):
            kwargs["device_map"] = "auto"
            return from_pretrained_original(*args, **kwargs)

        def hf_hub_download(repo_id: str, filename: str, **kwargs):
            if os.path.exists(os.path.join(repo_id, filename)):
                return os.path.join(repo_id, filename)
            return hf_hub_download_original(
                repo_id=repo_id, filename=filename, **kwargs
            )

        with patch(
            "text_generation_server.models.causal_lm.AutoModelForCausalLM.from_pretrained",
            from_pretrained,
        ), patch(
            "text_generation_server.models.rw.AutoModelForCausalLM.from_pretrained",
            from_pretrained,
        ), patch(  # remove after huggingface/text-generation-inference/pull/534 is merged
            "text_generation_server.models.mpt.hf_hub_download",
            hf_hub_download,
        ):
            super().__init__(
                lambda: get_model(
                    model_id=model_id,
                    revision=revision,
                    sharded=int(os.getenv("WORLD_SIZE", "1")) > 1
                    if sharded is None
                    else sharded,
                    quantize=quantize,
                    dtype=dtype,
                    trust_remote_code=trust_remote_code,
                )
            )

    def requires_padding(self) -> bool:
        return self._model.requires_padding
