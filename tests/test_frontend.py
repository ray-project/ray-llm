from aviary.frontend.app import do_query


def test_do_query():
    llm = "hf-internal-testing/tiny-random-gpt2"
    prompt = "test query"
    result = do_query(prompt, llm, llm, llm)
    assert result
