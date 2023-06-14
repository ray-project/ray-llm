from aviary.api import sdk


def test_get_backend():
    backend = sdk.get_aviary_backend()
    assert backend


def test_get_aviary():

    models = sdk.models()
    assert len(models) > 8
    assert "mosaicml/mpt-7b-instruct" in models

    completions = sdk.completions(model="mosaicml/mpt-7b-instruct", prompt="test")
    assert completions
    assert "generated_text" in completions.keys()

    batch_completions = sdk.batch_completions(
        model="mosaicml/mpt-7b-instruct",
        prompts=["test", "test"]
    )
    assert all(
        "generated_text" in batch_completions[i] for i in range(len(batch_completions))
    )


def test_list_models():
    all_models = sdk.models()

    assert len(all_models) > 8
    assert "mosaicml/mpt-7b-instruct" in all_models


def test_metadata():
    llm = "amazon/LightGPT"
    result = sdk.metadata(llm)
    assert "metadata" in result.keys()


def test_completions():
    llm = "amazon/LightGPT"
    prompt = "test query"
    result = sdk.completions(llm, prompt)
    assert result


def test_query():
    llm = "amazon/LightGPT"
    prompt = "test query"
    result = sdk.query(llm, prompt)
    assert result


def test_batch_query():
    llm = "amazon/LightGPT"
    prompts = ["test batch query", "test batch query 2"]
    result = sdk.batch_query(llm, prompts)
    assert result


def test_batch_completions():
    llm = "amazon/LightGPT"
    prompts = ["test batch query", "test batch query 2"]
    result = sdk.batch_completions(llm, prompts)
    assert result
