from . import mock_sdk


def test_get_aviary():
    models = mock_sdk.models()
    assert len(models) == 3

    completions = mock_sdk.completions(model="foo", prompt="test")
    assert completions
    assert "generated_text" in completions.keys()

    batch_completions = mock_sdk.batch_completions(
        model="mosaicml/mpt-7b-instruct", prompts=["test", "test"]
    )
    assert all(
        "generated_text" in batch_completions[i] for i in range(len(batch_completions))
    )


def test_list_models():
    all_models = mock_sdk.models()

    assert len(all_models) == 3


def test_metadata():
    llm = "bar"
    result = mock_sdk.metadata(llm)
    assert "metadata" in result.keys()


def test_query():
    llm = "baz"
    prompt = "test query"
    result = mock_sdk.completions(llm, prompt)
    assert result


def test_batch_query():
    llm = "foobar"
    prompts = ["test batch query", "test batch query 2"]
    result = mock_sdk.batch_completions(llm, prompts)
    assert result


def test_stream():
    llm = "bar"
    prompt = "test query"
    result = []
    for chunk in mock_sdk.stream(llm, prompt):
        result.append(chunk)
    assert result
