from aviary import sdk


class TestAviarySDK:
    """Test the core functions of the Aviary SDK."""

    def test_get_backend(self, aviary_testing_model):  # noqa: F811
        backend = sdk.get_aviary_backend()
        assert backend

    def test_get_aviary(self, aviary_testing_model):  # noqa: F811
        completions = sdk.completions(model=aviary_testing_model, prompt="test")
        assert completions

    def test_list_models(self, aviary_testing_model):  # noqa: F811
        all_models = sdk.models()

        assert len(all_models)
        assert aviary_testing_model in all_models

    def test_metadata(self, aviary_testing_model):  # noqa: F811
        result = sdk.metadata(aviary_testing_model)
        assert "aviary_metadata" in result.keys()

    def test_completions(self, aviary_testing_model):  # noqa: F811
        prompt = "test query"
        result = sdk.completions(aviary_testing_model, prompt)
        assert result

    def test_query(self, aviary_testing_model):  # noqa: F811
        prompt = "test query"
        result = sdk.query(aviary_testing_model, prompt)
        assert result["choices"][0]["message"]["content"]
        assert result["usage"]

    def test_stream(self, aviary_testing_model):  # noqa: F811
        prompt = "test query"
        for chunk in sdk.stream(aviary_testing_model, prompt):
            assert chunk["choices"][0]["delta"] or chunk["choices"][0]["finish_reason"]
        assert chunk["usage"]
