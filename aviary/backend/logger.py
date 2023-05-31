import logging
import os
from typing import Optional

LOG_FORMAT = (
    "[%(levelname)s %(asctime)s]{rank} %(filename)s: %(lineno)d  " "%(message)s"
)


def get_logger(name: str = None, rank: Optional[int] = None, **kwargs):
    if rank is None:
        rank = int(os.environ.get("RANK", -1))
    logger = logging.getLogger(name)
    level = logging.ERROR if rank > 0 else logging.INFO
    log_format = LOG_FORMAT.format(rank=f"[Rank {rank}]" if rank > -1 else "")
    logging.basicConfig(level=level, format=log_format, **kwargs)
    return logger
