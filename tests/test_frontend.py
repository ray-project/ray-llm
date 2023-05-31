from aviary.frontend.app import do_query


def test_do_query():
    llm = "amazon/LightGPT"
    prompt = "test query"
    result = do_query(prompt, llm, llm, llm)
    assert result
