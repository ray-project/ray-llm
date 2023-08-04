class TestFrontend:
    def test_do_query(self, aviary_testing_model):  # noqa: F811
        from aviary.frontend.app import do_query

        llm = aviary_testing_model
        prompt = "test query"
        result = do_query(prompt, llm, llm, llm)
        assert result
