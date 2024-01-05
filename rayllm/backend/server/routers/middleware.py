import secrets

from fastapi import Request

from rayllm.backend.observability.tracing.baggage import baggage


async def add_request_id(request: Request, call_next):
    request.state.request_id = secrets.token_urlsafe()

    with baggage({"request_id": request.state.request_id}):
        resp = await call_next(request)

    return resp
