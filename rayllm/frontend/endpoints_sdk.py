import os
from collections import namedtuple
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


def get_openai_client() -> Optional[openai.Client]:
    """Get an OpenAI Client connected to the Endpoints backend."""
    backend = get_endpoints_backend()
    if not backend.backend_url or not backend.token:
        return None
    return openai.Client(base_url=backend.backend_url, api_key=backend.token)


def get_endpoints_models() -> List[str]:
    """List available models"""
    openai_client = get_openai_client()
    if not openai_client:
        return []
    models = openai_client.models.list()
    return [model.id for model in models.data]


def get_endpoints_metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    openai_client = get_openai_client()
    if not openai_client:
        return {}
    metadata = openai_client.models.retrieve(model_id).model_dump()
    return metadata
