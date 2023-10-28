import asyncio

import pytest
from fastapi.datastructures import State

from rayllm.backend.observability import request_context


async def nested(state_value, level=5):
    """Assert that the state value stays constant through nested calls"""

    if level == 0:
        return

    assert request_context.get_fastapi_state() == state_value
    await asyncio.sleep(0.1)
    await nested(state_value, level - 1)


def test_request_context_fastapi():
    state = State()
    state.id = "id"
    assert request_context.maybe_get_string_field("id") is None
    assert request_context.get_fastapi_state() is None
    with request_context.set_fastapi_state(state):
        assert request_context.get_fastapi_state() == state
        assert request_context.get_fastapi_state().id == "id"
        assert request_context.maybe_get_string_field("id") == "id"

    assert request_context.get_fastapi_state() is None
    assert request_context.maybe_get_string_field("id") is None


def test_request_context():
    assert request_context.get("key") is None
    with request_context.set(key="hello"):
        # Request contexts don't affect each other
        assert request_context.get_fastapi_state() is None
        assert request_context.maybe_get_string_field("id") is None
        assert request_context.get("key") == "hello"
        with request_context.set(key="goodbye", other="new"):
            assert request_context.get("key") == "goodbye"
            assert request_context.get("other") == "new"
        assert request_context.get("key") == "hello"
        assert request_context.get("other") is None
    assert request_context.get("key") is None
    assert request_context.get("other") is None


@pytest.mark.asyncio
async def test_nested_request_context():
    state = State()
    state.id = "id"
    with request_context.set_fastapi_state(state):
        await nested(state)


@pytest.mark.asyncio
async def test_nested_multi_request_context():
    state = State()
    state.id = "id"
    with request_context.set_fastapi_state(state):
        t1 = asyncio.create_task(nested(state))
        new_state = State()
        new_state.id = "new_id"
        with request_context.set_fastapi_state(new_state):
            t2 = asyncio.create_task(nested(new_state))
            t3 = asyncio.create_task(nested(state))

    assert request_context.get_fastapi_state() is None
    await asyncio.gather(t1, t2)

    with pytest.raises(AssertionError):
        await t3
