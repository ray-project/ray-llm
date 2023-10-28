import pytest
from pydantic import ValidationError

from rayllm.backend.server.models import SamplingParams
from rayllm.env_conf import MAX_NUM_STOPPING_SEQUENCES


def test_sampling_params():
    assert SamplingParams(stop=None).stop is None
    assert SamplingParams(stop=[]).stop == []
    assert SamplingParams(stop=["stop"]).stop == ["stop"]
    assert SamplingParams(stop=["stop", "stop", "stop2"]).stop == ["stop", "stop2"]
    assert SamplingParams(stop=["stop"] * 100).stop == ["stop"]
    max_len_list = [str(i) for i in range(MAX_NUM_STOPPING_SEQUENCES)]
    assert SamplingParams(stop=max_len_list).stop == max_len_list

    with pytest.raises(ValidationError):
        SamplingParams(stop=max_len_list + ["too many"])
