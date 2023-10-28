import os
from collections import namedtuple
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import openai

from rayllm.sdk import AviaryResource

__all__ = ["get_endpoints_backend", "get_endpoints_models", "get_endpoints_metadata"]

response = namedtuple("Response", ["text", "status_code"])


def get_endpoints_backend(verbose: Optional[bool] = None) -> AviaryResource:
    """Construct a context for interacting with Anyscale Endpoints.

    Returns:
        An AviaryResource configured with the Anyscale Endpoints url and an endpoints
            token.
    """
    endpoints_url = os.getenv("ENDPOINTS_URL") or ""
    endpoints_token = os.getenv("ENDPOINTS_KEY") or ""

    if endpoints_url:
        endpoints_url += "/v1" if not endpoints_url.endswith("/v1") else ""

    if verbose is None:
        verbose = os.environ.get("AVIARY_SILENT", "0") == "0"
    if verbose:
        print(f"Connecting to Endpoints backend at: {endpoints_url}")
    return AviaryResource(endpoints_url, endpoints_token)


@contextmanager
def openai_endpoints_context():
    backend = get_endpoints_backend()
    original_api_base = openai.api_base
    original_api_key = openai.api_key
    openai.api_base = backend.backend_url
    openai.api_key = backend.token
    yield
    openai.api_base = original_api_base
    openai.api_key = original_api_key


@openai_endpoints_context()
def get_endpoints_models() -> List[str]:
    """List available models"""
    if not (openai.api_base and openai.api_key):
        return []
    models = openai.Model.list()
    return [model.id for model in models.data]


@openai_endpoints_context()
def get_endpoints_metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    if not (openai.api_base and openai.api_key):
        return {}
    metadata = openai.Model.retrieve(model_id)
    return metadata
