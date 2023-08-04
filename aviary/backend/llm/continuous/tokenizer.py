from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, List

import ray


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def __contains__(self, item: Any) -> bool:
        return item in self.cache

    def get(self, key: int) -> int:
        if key not in self.cache:
            raise KeyError()
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class Tokenizer(ABC):
    @abstractmethod
    def get_input_length(self, input_text: str) -> int:
        raise NotImplementedError("")


class NaiveTokenizer(Tokenizer):
    def get_input_length(self, input_text: str) -> int:
        return min(input_text.count(" ") + 1)


class TransformersTokenizer(Tokenizer):
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def get_input_length(self, input_text: str) -> int:
        return self._tokenizer(
            text=input_text,
            return_tensors="pt",
            return_token_type_ids=False,
        )["input_ids"].shape[1]


class RayTokenizer(Tokenizer):
    def __init__(self, worker_group: List[ray.ObjectRef]):
        self._worker_group = worker_group
        self._id = -1

    def get_input_length(self, input_text: str) -> ray.ObjectRef:
        self._id += 1
        # Simple round robin
        return self._worker_group[
            self._id % len(self._worker_group)
        ].get_input_length.remote(input_text)


class CachingTokenizer(Tokenizer):
    def __init__(self, tokenizer: Tokenizer, capacity: int = 128) -> None:
        self.tokenizer = tokenizer
        self._cache = LRUCache(capacity=capacity)

    def get_input_length(self, input_text: str) -> int:
        try:
            return self._cache.get(input_text)
        except KeyError:
            length = self.tokenizer.get_input_length(input_text)
            self._cache.put(input_text, length)
            return length
