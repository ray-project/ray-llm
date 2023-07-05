from aviary import sdk

TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"


def test_get_backend():
    backend = sdk.get_aviary_backend()
    assert backend


def test_get_aviary():
    completions = sdk.completions(model=TEST_MODEL, prompt="test")
    assert completions

    batch_completions = sdk.batch_completions(
        model=TEST_MODEL, prompts=["test", "test"]
    )
    assert batch_completions


def test_list_models():
    all_models = sdk.models()

    assert len(all_models)
    assert TEST_MODEL in all_models


def test_metadata():
    result = sdk.metadata(TEST_MODEL)
    assert "metadata" in result.keys()


def test_completions():
    prompt = "test query"
    result = sdk.completions(TEST_MODEL, prompt)
    assert result


def test_query():
    prompt = "test query"
    result = sdk.query(TEST_MODEL, prompt)
    assert result


def test_batch_query():
    prompts = ["test batch query", "test batch query 2"]
    result = sdk.batch_query(TEST_MODEL, prompts)
    assert result


def test_batch_completions():
    prompts = ["test batch query", "test batch query 2"]
    result = sdk.batch_completions(TEST_MODEL, prompts)
    assert result


def test_stream():
    prompt = "test query"
    result = []
    for chunk in sdk.stream(TEST_MODEL, prompt):
        result.append(chunk)
    assert result
