from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Request:
    id: int
    inputs: str
    truncate: int
    max_new_tokens: int
    params: Dict[str, Any]

    @property
    def batch_id(self) -> int:
        return self.id


__all__ = ["Request"]
