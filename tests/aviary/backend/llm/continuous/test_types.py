from aviary.backend.llm.continuous.types import TGIParams


def test_tgi_params():
    """Test the value checking on the TGIParams class."""
    params_dict = dict(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=2,
        top_p=0.1,
        typical_p=0.1,
        do_sample=True,
        stop_sequences=[],
        ignore_eos_token=False,
        watermark=False,
        seed=None,
        # Aviary params
        # Force at least one token to be generated.
        min_new_tokens=1,
        # OpenAI repetition penalties
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    params = TGIParams(**params_dict)
    # the params should remain unmodified after construction and temperature is greater
    # than 0
    for param in params_dict:
        assert getattr(params, param) == params_dict[param], (
            f"Expected {param} to be unmodified {params_dict[param]}, but got "
            f"{getattr(params, param)} instead."
        )

    # test that when temperature is 0, that the other sampling parameters are properly
    # set to disable any type of sampling, like in OpenAI API
    expected_modified_params = dict(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
        do_sample=False,
    )

    params_dict["temperature"] = 0.0

    temperature_zero_params = TGIParams(**params_dict)
    for param in expected_modified_params:
        assert (
            getattr(temperature_zero_params, param) == expected_modified_params[param]
        ), (
            f"Expected {param} to be modified: {expected_modified_params[param]}, but got "
            f"{getattr(temperature_zero_params, param)} instead."
        )

    # Test that None values are converted to numerical equivalents
    params_dict = dict(
        temperature=1.0,
        repetition_penalty=1.0,
        top_k=None,
        top_p=None,
        typical_p=None,
        do_sample=True,
        stop_sequences=[],
        ignore_eos_token=False,
        watermark=False,
        seed=None,
        # Aviary params
        # Force at least one token to be generated.
        min_new_tokens=1,
        # OpenAI repetition penalties
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    params = TGIParams(**params_dict)
    expected_params = dict(
        top_k=0,
        top_p=1.0,
        typical_p=1.0,
    )

    for param in expected_params:
        assert getattr(params, param) == expected_params[param], (
            f"Expected {param} to be {expected_params[param]}, but got "
            f"{getattr(params, param)} instead."
        )
