from typing import Dict, Union

from ray.util import metrics


class NonExceptionThrowingCounter(metrics.Counter):
    def inc(self, value: Union[int, float] = 1.0, tags: Dict[str, str] = None):
        if value > 0:
            return super().inc(value, tags)
