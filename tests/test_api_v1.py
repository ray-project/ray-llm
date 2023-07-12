import aviary as openai

TEST_MODEL = "hf-internal-testing/tiny-random-gpt2"


def test_models():
    models = openai.Model.list()
    print(models)

    assert type(models.data) == list
    print(models.data[0].id)


def test_completions():
    completion = openai.Completion.create(model=TEST_MODEL, prompt="Hello world")
    assert completion
    assert completion.usage
    assert completion.id
    assert type(completion.choices) == list
    assert completion.choices[0].text
    print(completion.choices[0].text)


def test_chat():
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": "Hello world"}]
    )
    assert chat_completion
    assert chat_completion.usage
    assert chat_completion.id
    assert type(chat_completion.choices) == list
    assert chat_completion.choices[0].message.content

    # print the chat completion
    print(chat_completion.choices[0].message.content)


def test_completions_stream():
    for completion in openai.Completion.create(
        model=TEST_MODEL, prompt="Hello world", stream=True
    ):
        assert completion
        assert completion.usage
        assert completion.id
        assert type(completion.choices) == list
        assert completion.choices[0].text
        print(completion.choices[0].text)


def test_chat_stream():
    # create a chat completion
    for chat_completion in openai.ChatCompletion.create(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "Hello world"}],
        stream=True,
    ):
        assert chat_completion
        assert chat_completion.usage
        assert chat_completion.id
        assert type(chat_completion.choices) == list
        assert chat_completion.choices[0].message.content

        # print the chat completion
        print(chat_completion.choices[0].message.content)
