import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Generic, List, Optional, TypeVar, Union

from rayllm.backend.llm.error_handling import InputTooLong

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int):
        self.cache: OrderedDict[K, V] = OrderedDict()
        self.capacity = capacity

    def __contains__(self, item: K) -> bool:
        return item in self.cache

    def get(self, key: K) -> V:
        if key not in self.cache:
            raise KeyError()
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: K, value: V) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class Tokenizer(ABC):
    def get_input_length(self, input_text: str) -> int:
        return len(self.encode(input_text))

    def validate_input_length(self, prompt_token_ids: List[int], max_input_length: int):
        num_input_tokens = len(prompt_token_ids)
        if num_input_tokens > max_input_length:
            logger.info("Task is over the max input length.")
            InputTooLong(num_input_tokens, max_input_length).raise_exception()
        return prompt_token_ids

    def encode_if_required(
        self,
        input_text_or_ids: Union[str, List[int]],
        max_input_length: Optional[int] = None,
    ):
        if isinstance(input_text_or_ids, str):
            input_ids = self.encode(input_text_or_ids)
        else:
            input_ids = input_text_or_ids

        # Validate if required
        return (
            self.validate_input_length(input_ids, max_input_length)
            if max_input_length
            else input_ids
        )

    @abstractmethod
    def encode(self, input_text: str) -> List[int]:
        raise NotImplementedError("")


class TransformersTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, input_text: str) -> List[int]:
        return self._tokenizer(
            text=input_text,
            return_token_type_ids=False,
        )["input_ids"]


class CachingTokenizer(Tokenizer):
    def __init__(self, tokenizer: Tokenizer, capacity: int = 4096) -> None:
        self.tokenizer = tokenizer
        self._cache = LRUCache[str, List[int]](capacity=capacity)

    def encode(self, input_text: str) -> List[int]:
        try:
            return self._cache.get(input_text)
        except KeyError:
            tokens = self.tokenizer.encode(input_text)
            self._cache.put(input_text, tokens)
            return tokens
