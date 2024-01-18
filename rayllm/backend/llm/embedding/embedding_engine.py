import asyncio
import logging
import time
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from rayllm.backend.llm.embedding.embedding_model_runner import get_model_runner
from rayllm.backend.llm.embedding.embedding_models import EmbeddingApp
from rayllm.backend.llm.error_handling import InputTooLong
from rayllm.backend.llm.llm_node_initializer import LLMNodeInitializer
from rayllm.backend.observability.base import step
from rayllm.backend.server.models import (
    AviaryModelResponse,
    GenerationRequest,
)
from rayllm.backend.server.utils import make_async

logger = logging.getLogger(__name__)


def _encode_for_embeddings(
    tokenizer: PreTrainedTokenizerBase, max_total_tokens: int, prompts: List[str]
) -> Tuple[dict, List[int]]:
    batch_dict = tokenizer(
        prompts,
        max_length=max_total_tokens + 1,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # If the second dimension of the output tensor is bigger
    # than the max_total_tokens, it means at least one request
    # is too long.
    if batch_dict["input_ids"].shape[1] > max_total_tokens:
        # We know that too long requests will have a non-padding
        # token in the last position of the input_ids tensor.
        is_too_long_mask: torch.Tensor = (
            batch_dict["input_ids"][:, -1] != tokenizer.pad_token_id
        )
        too_long_requests = is_too_long_mask.nonzero().flatten()
        ok_requests = is_too_long_mask.logical_not_().nonzero().flatten()
        # Keep just the requests that aren't too long.
        batch_dict = {k: t[ok_requests] for k, t in batch_dict.items()}
    else:
        ok_requests = torch.arange(len(prompts))
        too_long_requests = None

    # Get the actual number of tokens for each request (without padding).
    padding_mask: torch.Tensor = batch_dict["input_ids"] != tokenizer.pad_token_id
    num_tokens = padding_mask.sum(dim=1)
    num_tokens_all = torch.empty(len(prompts), dtype=num_tokens.dtype)
    if ok_requests is not None:
        num_tokens_all.index_copy_(0, ok_requests, num_tokens)
    if too_long_requests is not None:
        num_tokens_all.index_fill_(0, too_long_requests, -1)

    # We need to truncate again here.
    # num_tokens.numel() will be 0 if all prompts are too long.
    if num_tokens.numel() > 0:
        truncate_index = num_tokens.max()
        batch_dict = {k: t[:, :truncate_index] for k, t in batch_dict.items()}

    return batch_dict, num_tokens_all.tolist()


class EmbeddingEngine:
    def __init__(
        self, llm_app: EmbeddingApp, *, node_initializer: LLMNodeInitializer = None
    ):
        """Initialize the tokenizer and model
        Args:
            llm_app: the configuration for this engine
            node_initializer: node initializer
        """
        self.llm_app = llm_app.copy(deep=True)
        self.max_total_tokens = llm_app.engine_config.max_total_tokens

        self.node_initializer = node_initializer or LLMNodeInitializer(
            local_node_tokenizer_only=False
        )
        self.running = False
        self._execution_lock = asyncio.Lock()

    def _get_tokenizer(self):
        # get tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_app.engine_config.tokenizer_id
            or self.llm_app.engine_config.actual_hf_model_id
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_model(self):
        # get model
        return get_model_runner(self.llm_app)

    def _warmup(self):
        max_batch_size = self.llm_app.engine_config.max_batch_size
        max_total_tokens = self.llm_app.engine_config.max_total_tokens
        logger.info(f"Warming up with {max_batch_size}*{max_total_tokens} tokens...")
        input_data = torch.tensor(
            [[1] * max_total_tokens] * max_batch_size, dtype=torch.long
        )
        batch_dict = {
            "input_ids": input_data,
            "attention_mask": input_data.clone(),
            "token_type_ids": input_data.clone(),
        }
        t = time.perf_counter()
        self.model(batch_dict)
        et = time.perf_counter() - t
        logger.info(
            f"Warmup complete in {et}s (throughput: {max_batch_size*max_total_tokens/et} tokens/s)."
        )

    @make_async
    def _encode_for_embeddings_async(self, *args, **kwargs):
        return _encode_for_embeddings(
            self.tokenizer, self.max_total_tokens, *args, **kwargs
        )

    @make_async
    def _run_model_async(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    async def start(self):
        """Start the Embedding Engine

        If the engine is already running, do nothing.

        """
        if self.running:
            # The engine is already running!
            logger.info("Skipping engine restart because the engine is already running")
            return

        # Get the scaling options
        with step("Starting embedding engine", request_id="node_initialize"):
            pg, runtime_env = await self.node_initializer.initialize_node(self.llm_app)

        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()
        self._warmup()

        self.running = True

    async def generate(
        self, engine_request: GenerationRequest, normalization: bool = True
    ):
        # Tokenize the input texts
        prompts = [str(x) for x in engine_request.prompt]
        logger.info(
            f"Starting encoding for ({len(engine_request.request_id)}) {','.join(rid for rid in engine_request.request_id)}"
        )
        async with self._execution_lock:
            batch_dict, num_tokens_all = await self._encode_for_embeddings_async(
                prompts
            )

            if batch_dict["input_ids"].numel() > 0:
                logger.info(
                    f"Starting model forward pass for ({len(engine_request.request_id)}) {','.join(rid for rid in engine_request.request_id)}"
                )
                t = await self._run_model_async(batch_dict, normalization)
            else:
                logger.info(
                    f"No valid requests for ({len(engine_request.request_id)}) {','.join(rid for rid in engine_request.request_id)}"
                )

        batch_dict_idx = 0
        responses = []
        for num_tokens in num_tokens_all:
            if num_tokens > -1:
                responses.append(
                    AviaryModelResponse(
                        embedding_outputs=t[batch_dict_idx].tolist(),
                        num_input_tokens=num_tokens,
                        num_input_tokens_batch=num_tokens,
                    )
                )
                batch_dict_idx += 1
            else:
                # Do not raise here
                exc = InputTooLong(-1, self.max_total_tokens).exception
                responses.append(exc)
        logger.info(
            f"Yielding responses for ({len(engine_request.request_id)}) {','.join(rid for rid in engine_request.request_id)}"
        )
        yield responses

    async def check_health(self) -> bool:
        return True

    def stats(self):
        pass

    def shutdown(self):
        raise NotImplementedError()
