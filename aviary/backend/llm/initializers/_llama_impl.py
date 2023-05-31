import sys
import time
import uuid
from typing import Dict, Iterator, List, Optional, Tuple, Union

from llama_cpp import Completion, CompletionChunk, CompletionLogprobs, Llama, llama_cpp

from aviary.backend.logger import get_logger

logger = get_logger(__name__)

# ruff: noqa: B006, B905


# TODO Upstream this
# Llama with added min_tokens parameter
class LlamaWithMinLen(Llama):
    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        min_tokens: int = 0,  # new aviary argument
        max_time_criteria: Optional[Tuple[float, float]] = None,  # new aviary argument
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[str]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        return self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            max_time_criteria=max_time_criteria,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
        )

    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        min_tokens: int = 0,
        max_time_criteria: Optional[Tuple[float, float]] = None,  # new aviary argument
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[str]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            max_time_criteria=max_time_criteria,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
        )
        if stream:
            chunks: Iterator[CompletionChunk] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    def _create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 16,
        min_tokens: int = 0,
        max_time_criteria: Optional[Tuple[float, float]] = None,  # new aviary argument
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[List[str]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
    ) -> Union[Iterator[Completion], Iterator[CompletionChunk]]:
        assert self.ctx is not None
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        completion_tokens: List[llama_cpp.llama_token] = []
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[llama_cpp.llama_token] = self.tokenize(
            b" " + prompt.encode("utf-8")
        )
        text: bytes = b""
        returned_characters: int = 0
        stop = stop if stop is not None else []

        if self.verbose:
            llama_cpp.llama_reset_timings(self.ctx)

        if len(prompt_tokens) + max_tokens > int(llama_cpp.llama_n_ctx(self.ctx)):
            raise ValueError(
                f"Requested tokens exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self.params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache:
            try:
                cache_item = self.cache[prompt_tokens]
                cache_prefix_len = Llama.longest_token_prefix(
                    cache_item.eval_tokens, prompt_tokens
                )
                eval_prefix_len = Llama.longest_token_prefix(
                    self.eval_tokens, prompt_tokens
                )
                if cache_prefix_len > eval_prefix_len:
                    self.load_state(cache_item)
                    if self.verbose:
                        print("Llama._create_completion: cache hit", file=sys.stderr)
            except KeyError:
                if self.verbose:
                    print("Llama._create_completion: cache miss", file=sys.stderr)

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
        ):
            if max_time_criteria:
                max_time, initial_timestamp = max_time_criteria
                if time.time() - initial_timestamp > max_time:
                    finish_reason = "time"
                    break

            if token == llama_cpp.llama_token_eos():
                if len(completion_tokens) >= min_tokens:
                    text = self.detokenize(completion_tokens)
                    finish_reason = "stop"
                    break
            else:
                completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens)

            # Contains multi-byte UTF8
            for k, char in enumerate(all_text[-3:]):
                k = 3 - k
                for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                    # Bitwise AND check
                    if num > k and pattern & char == pattern:
                        multibyte_fix = num - k

            # Stop incomplete bytes from passing
            if multibyte_fix > 0:
                multibyte_fix -= 1
                continue

            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                start = returned_characters
                longest = 0
                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                for s in stop_sequences:
                    for i in range(len(s), 0, -1):
                        if all_text.endswith(s[:i]):
                            if i > longest:
                                longest = i
                            break
                text = all_text[: len(all_text) - longest]
                returned_characters += len(text[start:])
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_path,
                    "choices": [
                        {
                            "text": text[start:].decode("utf-8", errors="ignore"),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        if self.verbose:
            llama_cpp.llama_print_timings(self.ctx)

        if stream:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": self.model_path,
                "choices": [
                    {
                        "text": text[returned_characters:].decode(
                            "utf-8", errors="ignore"
                        ),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            return

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0
            text_offsets: List[int] = []
            token_logprobs: List[float] = []
            tokens: List[str] = []
            top_logprobs: List[Dict[str, float]] = []

            all_tokens = prompt_tokens + completion_tokens
            all_token_strs = [
                self.detokenize([token]).decode("utf-8", errors="ignore")
                for token in all_tokens
            ]
            all_logprobs = [
                Llama.logits_to_logprobs(list(map(float, row)))
                for row in self.eval_logits
            ]
            for token, token_str, logprobs_token in zip(
                all_tokens, all_token_strs, all_logprobs
            ):
                text_offsets.append(text_offset)
                text_offset += len(token_str)
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(sorted_logprobs[int(token)][0])
                top_logprob = {
                    self.detokenize([llama_cpp.llama_token(i)]).decode(
                        "utf-8", errors="ignore"
                    ): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: sorted_logprobs[int(token)][0]})
                top_logprobs.append(top_logprob)
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_path,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }
