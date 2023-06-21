import aviary as openai


def test_models():
    models = openai.Model.list()
    print(models)

    assert type(models.data) == list
    print(models.data[0].id)


def test_completions():
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt="Hello world"
    )
    assert completion
    assert completion.usage
    assert completion.id
    assert type(completion.choices) == list
    assert completion.choices[0].text
    print(completion.choices[0].text)


def test_chat():
    # create a chat completion
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello world"}]
    )
    assert chat_completion
    assert chat_completion.usage
    assert chat_completion.id
    assert type(chat_completion.choices) == list
    assert chat_completion.choices[0].message.content

    # print the chat completion
    print(chat_completion.choices[0].message.content)
