import asyncio
import os
import subprocess
import time
import traceback
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest.mock import patch

import requests
import torch
import torch.distributed as dist
from filelock import FileLock
from ray.air.util.torch_dist import (
    ActorHandle,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    get_address_and_port,
)
from torch.hub import _get_torch_home
from transformers import PreTrainedTokenizer

from rayllm.backend.logger import get_logger
from rayllm.backend.server.models import (
    GCSMirrorConfig,
    S3AWSCredentials,
    S3MirrorConfig,
)

T = TypeVar("T")
logger = get_logger(__name__)
AWS_EXECUTABLE = "aws"


def download_model_from_s3(
    model_id: str,
    path: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    tokenizer_only: bool = False,
    aws_executable: str = AWS_EXECUTABLE,
    env: Optional[Dict[str, str]] = None,
) -> None:
    """
    Download a model from an S3 bucket and save it in TRANSFORMERS_CACHE for
    seamless interoperability with Hugging Face's Transformers library.

    The downloaded model may have a 'hash' file containing the commit hash
    corresponding to the commit on Hugging Face Hub.
    """
    s3_sync_args = s3_sync_args or []
    subprocess.run(
        [aws_executable, "s3", "cp", "--quiet"]
        + s3_sync_args
        + [os.path.join(bucket_uri, "hash"), "."],
        env=env,
    )
    if not os.path.exists(os.path.join(".", "hash")):
        f_hash = "0000000000000000000000000000000000000000"
        logger.warning(
            f"hash file does not exist in {bucket_uri}. Using {f_hash} as the hash."
        )
    else:
        with open(os.path.join(".", "hash"), "r") as f:
            f_hash = f.read().strip()
    logger.info(
        f"Downloading {model_id} from {bucket_uri} to {os.path.join(path, 'snapshots', f_hash)}"
    )
    subprocess.run(["mkdir", "-p", os.path.join(path, "snapshots", f_hash)])
    subprocess.run(["mkdir", "-p", os.path.join(path, "refs")])
    subprocess.run(
        [
            aws_executable,
            "s3",
            "sync",
            "--quiet",
        ]
        + s3_sync_args
        + (
            ["--exclude", "*", "--include", "*token*", "--include", "config.json"]
            if tokenizer_only
            else []
        )
        + [
            bucket_uri,
            os.path.join(path, "snapshots", f_hash),
        ],
        env=env,
    )
    with open(os.path.join(path, "refs", "main"), "w") as f:
        f.write(f_hash)


def download_model_from_gcs(
    destination_path: str, bucket_uri: str, tokenizer_only: bool
) -> None:
    """
    Download a model from a GCS bucket and save it in TRANSFORMERS_CACHE for
    seamless interoperability with Hugging Face's Transformers library.

    The downloaded model may have a 'hash' file containing the commit hash corresponding
    to the commit on Hugging Face Hub. If not, we set the hash to a default
    value.

    The files are downloaded to the destination_path/snapshots/HASH/ directory.
    This function also writes a destination_path/refs/main file that contains
    the hash.

    Args:
        destination_path: The file path of the directory where all the files
            will be downloaded.
        bucket_uri: The URI of the GCS bucket to download files from.
        tokenizer_only: If True, only the files needed for the model's
            tokenizer will be downloaded.
    """

    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError(
            "You must `pip install google-cloud-storage` "
            "to download models from Google Cloud Storage."
        ) from e

    bucket_name, prefix = get_gcs_bucket_name_and_prefix(bucket_uri)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    logger.info(
        f'Downloading files from GCS bucket "{bucket_name}" at prefix ' f'"{prefix}".'
    )

    # Download hash file if it exists and get the hash. Otherwise, set
    # hash to a default.
    f_hash = "0000000000000000000000000000000000000000"
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name == f"{prefix}hash":
            blob.download_to_filename("./hash")
            with open(os.path.join(".", "hash"), "r") as f:
                f_hash = f.read().strip()
            logger.info(
                f"Detected hash file in GCS bucket {bucket_uri}. "
                f"Using {f_hash} as the hash."
            )
            break
    else:
        logger.warning(
            f"Hash file does not exist in GCS bucket {bucket_uri}. "
            f"Using {f_hash} as the hash."
        )

    # Write hash name to path/refs/main file.
    main_dir = os.path.join(destination_path, "refs")
    os.makedirs(main_dir, exist_ok=True)
    with open(os.path.join(main_dir, "main"), "w") as f:
        f.write(f_hash)

    destination_dir = os.path.join(destination_path, "snapshots", f_hash)
    os.makedirs(destination_dir, exist_ok=True)

    logger.info(f'Downloading model files to directory "{destination_dir}".')

    # Download all files in bucket to the path/snapshots/<f_hash>/ directory.
    # Blob names can contain slashes (/). However, GCS doesn't actually contain
    # true directories. We create the directories manually before downloading
    # blobs to mirror the directory structure in the bucket.
    tokenizer_file_substrings = ["tokenizer", "config.json"]
    for blob in bucket.list_blobs(prefix=prefix):
        # Remove the prefix from each blob's name
        blob_base_name = blob.name[len(prefix) :]

        if tokenizer_only:
            for substring in tokenizer_file_substrings:
                if substring not in blob_base_name:
                    continue
        if "/" in blob_base_name:
            blob_source_dir = blob_base_name[: blob_base_name.rfind("/")]
            blob_destination_dir = os.path.join(destination_dir, blob_source_dir)
            os.makedirs(blob_destination_dir, exist_ok=True)

        blob_destination_path = os.path.join(destination_dir, blob_base_name)
        blob.download_to_filename(blob_destination_path)


def get_gcs_bucket_name_and_prefix(bucket_uri: str) -> Tuple[str, str]:
    """Gets the GCS bucket name and prefix from the bucket_uri.

    The bucket name never includes a trailing slash.
    Any non-empty prefix always includes a trailing slash.
    """

    if not bucket_uri.startswith("gs://"):
        raise ValueError(
            f'Got invalid bucket_uri "{bucket_uri}". Expected a value that '
            'starts with "gs://".'
        )

    stripped_uri = bucket_uri[len("gs://") :]
    split_uri = stripped_uri.split("/", maxsplit=1)

    bucket_name = split_uri[0]

    if len(split_uri) > 1:
        bucket_prefix = split_uri[1]
    else:
        bucket_prefix = ""

    # Ensure non-empty bucket_prefixes have a trailing slash.
    if bucket_prefix != "" and not bucket_prefix.endswith("/"):
        bucket_prefix += "/"

    return bucket_name, bucket_prefix


def timeit(func):
    """
    Decorator to time a function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        logger.info(f"{func} took {time_taken} s to complete")
        return ret

    return inner


def get_aws_credentials(
    s3_aws_credentials_config: S3AWSCredentials,
) -> Optional[Dict[str, str]]:
    """
    This function creates temporary AWS credentials from a configured endpoint by issuing a POST request to the configured API.
    The function optionally uses an env variable for authorization and the returned result is a set of env variables that should
    be injected to the process issuing the S3 sync.
    """
    token = (
        os.getenv(s3_aws_credentials_config.auth_token_env_variable)
        if s3_aws_credentials_config.auth_token_env_variable
        else None
    )
    headers = {"Authorization": f"Bearer {token}"} if token else None
    resp = requests.post(
        s3_aws_credentials_config.create_aws_credentials_url, headers=headers
    )
    if not resp.ok:
        logger.error(f"Request to create AWS credentials had failed with {resp.reason}")
        return None

    env = resp.json()
    return env


def initialize_node(
    model_id: Optional[str] = None,
    s3_mirror_config: Optional[S3MirrorConfig] = None,
    gcs_mirror_config: Optional[GCSMirrorConfig] = None,
    tokenizer_only: bool = False,
) -> Optional[str]:
    """
    Perform initialization for a node.

    Currently, that means downloading the model from the S3 or GCS bucket.

    Returns path to downloaded model, if any.
    """

    # Create the torch cache kernels directory if it doesn't exist.
    # This is a workaround for a torch issue, where the kernels directory
    # cannot be created by torch if the parent directory doesn't exist.
    torch_cache_home = _get_torch_home()
    os.makedirs(os.path.join(torch_cache_home, "kernels"), exist_ok=True)

    if model_id is None:
        return None

    if s3_mirror_config is not None and gcs_mirror_config is not None:
        raise ValueError(
            "Received both s3_mirror_config and gcs_error_config. "
            "Please pass in only one config."
        )
    elif s3_mirror_config is not None:
        logger.info(
            "Received s3_mirror_config. Preparing to download model from AWS S3."
        )
        return get_model_from_s3(model_id, s3_mirror_config, tokenizer_only)
    elif gcs_mirror_config is not None:
        logger.info(
            "Received gcs_mirror_config. Preparing to download model from "
            "Google Cloud Storage."
        )
        return get_model_from_gcs(model_id, gcs_mirror_config, tokenizer_only)
    else:
        logger.info(
            "Did not receive s3_mirror_config or gcs_error_config. "
            "Not downloading model from AWS S3 or Google Cloud Storage."
        )
        return None


def get_model_from_s3(
    model_id: str, s3_mirror_config: S3MirrorConfig, tokenizer_only: bool
) -> Optional[str]:
    """Gets a model from S3 and stores it locally.

    Args:
        model_id: The HuggingFace ID of the model. Used only to construct the
            local destination file paths and for logging.
        s3_mirror_config: AWS S3 configuration.
        tokenizer_only: whether to download only the tokenizer files.

    Returns: file path of model if downloaded.
    """

    bucket_uri = s3_mirror_config.bucket_uri
    if bucket_uri is None:
        logger.info(
            "No bucket_uri was provided in the s3_mirror_config. "
            "Cannot download model from S3."
        )
        return None

    env_vars = None
    if s3_mirror_config.s3_aws_credentials is not None:
        env_vars = get_aws_credentials(s3_mirror_config.s3_aws_credentials)

    # TODO (shrekris-anyscale): add comment for why this is a delayed import.
    from transformers.utils.hub import TRANSFORMERS_CACHE

    path = os.path.expanduser(
        os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    )
    lock_path = os.path.expanduser(f"~/{model_id.replace('/', '--')}.lock")

    try:
        # Timeout 0 means there will be only one attempt to acquire
        # the file lock. If it cannot be aquired, a TimeoutError
        # will be thrown.
        # This ensures that subsequent processes don't duplicate work.
        with FileLock(lock_path, timeout=0):
            s3_sync_args = s3_mirror_config.s3_sync_args if s3_mirror_config else None
            try:
                download_model_from_s3(
                    model_id,
                    path,
                    bucket_uri,
                    s3_sync_args=s3_sync_args,
                    tokenizer_only=tokenizer_only,
                    env=env_vars,
                )
                logger.info("Done downloading the model from S3 bucket!")
            except RuntimeError:
                logger.warning(
                    "Unable to download the model from S3 bucket. "
                    f"Traceback:\n {traceback.format_exc()}"
                )
    except TimeoutError:
        # If the directory is already locked, then wait but do not do anything.
        with FileLock(lock_path, timeout=-1):
            pass
    return get_model_location_on_disk(model_id)


def get_model_from_gcs(
    model_id: str, gcs_mirror_config: GCSMirrorConfig, tokenizer_only: bool
) -> Optional[str]:
    """Gets a model from Google Cloud Storage and stores it locally.

    Args:
        model_id: The HuggingFace ID of the model. Used only to construct the
            local destination file paths and for logging.
        s3_mirror_config: GCS configuration.
        tokenizer_only: whether to download only the tokenizer files.

    Returns: file path of model if downloaded.
    """

    bucket_uri = gcs_mirror_config.bucket_uri
    if bucket_uri is None:
        logger.info(
            "No bucket_uri was provided in the "
            "gcs_mirror_config. Cannot download model from Google Cloud "
            "Storage."
        )
        return None

    # TODO (shrekris-anyscale): add comment for why this is a delayed import.
    from transformers.utils.hub import TRANSFORMERS_CACHE

    path = os.path.expanduser(
        os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    )
    lock_path = os.path.expanduser(f"~/{model_id.replace('/', '--')}.lock")

    try:
        # Timeout 0 means there will be only one attempt to acquire
        # the file lock. If it cannot be aquired, a TimeoutError
        # will be thrown.
        # This ensures that subsequent processes don't duplicate work.
        with FileLock(lock_path, timeout=0):
            try:
                download_model_from_gcs(
                    destination_path=path,
                    bucket_uri=bucket_uri,
                    tokenizer_only=tokenizer_only,
                )
                logger.info("Done downloading the model from GCS bucket!")
            except RuntimeError:
                logger.warning(
                    "Unable to download the model from GCS bucket. "
                    f"Traceback:\n {traceback.format_exc()}"
                )
    except TimeoutError:
        # If the directory is already locked, then wait but do not do anything.
        with FileLock(lock_path, timeout=-1):
            pass
    return get_model_location_on_disk(model_id)


def noop(*args, **kwargs):
    pass


def _init_torch_distributed_env_vars_only(*args, **kwargs):
    """Same as _init_torch_distributed, but only sets env vars."""
    with patch("torch.distributed.init_process_group", noop):
        _init_torch_distributed(*args, **kwargs)


async def init_torch_dist_process_group_async(
    workers: List[ActorHandle],
    backend: str = "gloo",
    init_method: str = "env",
    init_function: Callable = _init_torch_distributed,
) -> List[int]:
    """Initialize a torch distributed process group asynchronously.

    This is identical to
    ``ray.air.util.torch_dist.init_torch_dist_process_group``
    but uses asyncio to avoid blocking the event loop.

    Note: this util assumes that the order of the workers passed in
    are their global ranks.

    Args:
        workers: A list of TorchDistributedWorker actors.
        backend: The torch distributed backend to use,
            possible choices are "gloo" or "nccl".
        init_method: The initialization method to use,
            possible choices are "env" or "tcp".
        init_function: The function to use to initialize the
            torch distributed process group.

    Returns:
        Local ranks on their respective nodes for the list of workers.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed torch is not available.")

    # Build a map from node_id to workers on that node.
    node_and_gpu_ids = await asyncio.gather(
        *[w.execute.remote(_get_node_and_gpu_ids) for w in workers]
    )
    # All the workers on a specific node.
    node_to_workers = defaultdict(list)
    # All the gpu ids visible to all the workers on a specific node.
    node_to_gpu_ids = defaultdict(set)
    for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
        node_to_workers[node_id].append(i)
        # Force list.
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        # It is possible for a worker to have access to multiple GPUs.
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(gpu_id)

    # Assume the first worker is the master.
    master_addr, master_port = (
        await asyncio.gather(workers[0].execute.remote(get_address_and_port))
    )[0]

    setup_futures = []
    world_size = len(workers)
    local_ranks = []
    for rank, worker in enumerate(workers):
        node_id = node_and_gpu_ids[rank][0]
        local_rank = node_to_workers[node_id].index(rank)
        local_world_size = len(node_to_workers[node_id])
        setup_futures.append(
            worker.execute.remote(
                init_function,
                init_method=init_method,
                backend=backend,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                # list(set) will sort the gpu ids, so VISIBLE_CUDA_DEVICES
                # is always sorted.
                gpu_ids=list(node_to_gpu_ids[node_id]),
            )
        )
        local_ranks.append(local_rank)

    # Wait for all workers to join the process group.
    await asyncio.gather(*setup_futures)

    return local_ranks


def is_rank_zero():
    return int(os.environ.get("RANK", -1)) <= 0


def tokenize_string(tokenizer: PreTrainedTokenizer, key: str) -> Union[int, List[int]]:
    """Tokenize a string using a tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        key (str): String to tokenize.
    """
    token_ids = tokenizer.encode(key, add_special_tokens=False)
    return token_ids[0] if len(token_ids) == 1 else token_ids


def decode_tokens(tokenizer: PreTrainedTokenizer, tokens: Union[int, List[int]]) -> str:
    tokens = tokens if isinstance(tokens, list) else [tokens]
    text = tokenizer.decode(tokens)
    return text


def truncate_to_first_stop_token(
    tokens: torch.LongTensor,
    stop_ids: List[Union[int, List[int]]],
) -> torch.LongTensor:
    """Truncate tokens up to the first stop_id.

    Args:
        tokens (torch.LongTensor): Tokens to truncate.
        stop_ids (List[Union[int, List[int]]]): Stop ids to truncate at. Can be
            composed of single stop ids or sequences of ids.
    """
    if not stop_ids:
        return tokens
    stop_ids: List[torch.LongTensor] = [
        torch.LongTensor([stop_id] if not isinstance(stop_id, list) else stop_id)
        for stop_id in stop_ids
    ]
    for i in range(len(tokens)):
        for stop_id_index, _ in enumerate(stop_ids):
            stop_id = stop_ids[stop_id_index].to(tokens.device)
            if len(tokens) - i >= len(stop_id) and tokens[i : len(stop_id) + i].equal(
                stop_id
            ):
                return tokens[:i]
    return tokens


def tokenize_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[Union[List[int], int]]:
    """If any sequence is a string, tokenize it.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use.
        stopping_sequences (List[Union[str, int, List[int]]]): Stopping sequences to
            tokenize. Can be ids, sequences of ids or strings.
    """
    if not stopping_sequences:
        return None
    return [
        tokenize_string(tokenizer, sequence) if isinstance(sequence, str) else sequence
        for sequence in stopping_sequences
    ]


def decode_stopping_sequences_where_needed(
    tokenizer: PreTrainedTokenizer,
    stopping_sequences: List[Union[str, int, List[int]]],
) -> List[str]:
    """If any sequence is a string, tokenize it."""
    if not stopping_sequences:
        return None
    return [
        decode_tokens(tokenizer, sequence)
        if not isinstance(sequence, str)
        else sequence
        for sequence in stopping_sequences
    ]


def pythonize_tensors(obj: T) -> T:
    if isinstance(obj, dict):
        return {k: pythonize_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [pythonize_tensors(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(pythonize_tensors(v) for v in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    elif hasattr(obj, "__dict__"):
        obj.__dict__ = pythonize_tensors(obj.__dict__)
        return obj
    else:
        return obj


def get_model_location_on_disk(model_id: str) -> str:
    """Get the location of the model on disk.

    Args:
        model_id (str): Hugging Face model ID.
    """
    from transformers.utils.hub import TRANSFORMERS_CACHE

    model_dir = Path(
        TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}"
    ).expanduser()
    model_id_or_path = model_id

    model_dir_refs_main = Path(model_dir, "refs", "main")

    if model_dir.exists() and model_dir_refs_main.exists():
        with open(model_dir_refs_main, "r") as f:
            snapshot_hash = f.read().strip()

        snapshot_hash_path = Path(model_dir, "snapshots", snapshot_hash)
        if (
            snapshot_hash_path.exists()
            and Path(snapshot_hash_path, "config.json").exists()
        ):
            model_id_or_path = str(snapshot_hash_path.absolute())

    return model_id_or_path
