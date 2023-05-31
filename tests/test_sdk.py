from aviary.api.sdk import batch_query, models, query


def test_list_models():
    all_models = models()

    assert len(all_models) > 8
    assert "mosaicml/mpt-7b-instruct" in all_models


def test_query():
    llm = "amazon/LightGPT"
    prompt = "test query"
    result = query(llm, prompt)
    assert result


def test_batch_query():
    llm = "amazon/LightGPT"
    prompts = ["test batch query", "test batch query 2"]
    result = batch_query(llm, prompts)
    assert result
    print(result)
