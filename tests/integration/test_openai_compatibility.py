import openai
import openai.error
import pytest

from rayllm.sdk import openai_aviary_context


@pytest.fixture
def openai_testing_model(aviary_testing_model):
    with openai_aviary_context():
        yield aviary_testing_model


class TestOpenAICompatibility:
    """Test that the aviary endpoints are compatible with the OpenAI API"""

    def test_models(self, openai_testing_model):  # noqa: F811
        models = openai.Model.list()
        assert len(models["data"]) == 1, "Only the test model should be returned"
        assert (
            models.data[0].id == openai_testing_model
        ), "The test model id should match"

    def test_completions(self, openai_testing_model):  # noqa: F811
        completion = openai.Completion.create(
            model=openai_testing_model,
            prompt="Hello world",
            typical_p=0.1,
            max_tokens=2,
        )
        assert completion.model == openai_testing_model
        assert completion.model
        assert completion.choices[0].text == "test_0 test_1 "

    def test_chat(self, openai_testing_model):  # noqa: F811
        # create a chat completion
        chat_completion = openai.ChatCompletion.create(
            model=openai_testing_model,
            messages=[{"role": "user", "content": "Hello world"}],
            typical_p=1,
        )
        assert chat_completion
        assert chat_completion.usage
        assert chat_completion.id
        assert isinstance(chat_completion.choices, list)
        assert chat_completion.choices[0].message.content

    def test_completions_bad_request(self, openai_testing_model):  # noqa: F811
        with pytest.raises(openai.error.InvalidRequestError) as exc_info:
            openai.Completion.create(
                model=openai_testing_model,
                prompt="Hello world",
                temperature=-0.1,
            )
        assert "temperature" in str(exc_info.value)

    def test_chat_bad_request(self, openai_testing_model):  # noqa: F811
        with pytest.raises(openai.error.InvalidRequestError) as exc_info:
            openai.ChatCompletion.create(
                model=openai_testing_model,
                messages=[{"role": "user", "content": "Hello world"}],
                temperature=-0.1,
            )
        assert "temperature" in str(exc_info.value)

    def test_completions_stream(self, openai_testing_model):  # noqa: F811
        i = 0
        for completion in openai.Completion.create(
            model=openai_testing_model, prompt="Hello world", stream=True, typical_p=1
        ):
            i += 1
            assert completion
            assert completion.id
            assert isinstance(completion.choices, list)
            assert isinstance(completion.choices[0].text, str)
        assert i > 4

    def test_chat_stream(self, openai_testing_model):  # noqa: F811
        i = 0
        for chat_completion in openai.ChatCompletion.create(
            model=openai_testing_model,
            messages=[{"role": "user", "content": "Hello world"}],
            stream=True,
            temperature=0.4,
            frequency_penalty=0.02,
            typical_p=1,
        ):
            if i == 0:
                assert chat_completion
                assert chat_completion.id
                assert isinstance(chat_completion.choices, list)
                assert chat_completion.choices[0].delta.role
            else:
                assert chat_completion
                assert chat_completion.id
                assert isinstance(chat_completion.choices, list)
                assert chat_completion.choices[0].delta == {} or hasattr(
                    chat_completion.choices[0].delta, "content"
                )
            i += 1
        assert chat_completion
        assert chat_completion.id
        assert isinstance(chat_completion.choices, list)
        assert chat_completion.choices[0].delta == {}
        assert chat_completion.choices[0].finish_reason
        assert i > 4

    def test_completions_stream_bad_request(self, openai_testing_model):  # noqa: F811
        with pytest.raises(openai.error.APIError) as exc_info:
            for _ in openai.Completion.create(
                model=openai_testing_model,
                prompt="Hello world",
                stream=True,
                temperature=-0.1,
            ):
                pass
        assert "temperature" in str(exc_info.value)

    def test_chat_stream(self, openai_testing_model):  # noqa: F811
        with pytest.raises(openai.error.APIError) as exc_info:
            for _chat_completion in openai.ChatCompletion.create(
                model=openai_testing_model,
                messages=[{"role": "user", "content": "Hello world"}],
                stream=True,
                temperature=-0.1,
            ):
                pass
        assert "temperature" in str(exc_info.value)
