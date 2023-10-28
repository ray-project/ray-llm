import pytest
from pydantic import ValidationError
from transformers import AutoTokenizer

from rayllm.common.models import Message, Prompt, PromptFormat


def test_prompt_format_with_prompt_obj():
    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="",
    )
    prompt = prompt_format.generate_prompt(
        Prompt(
            prompt="hello1",
            use_prompt_format=True,
        )
    )
    assert prompt == "[user] hello1 [/user] [assistant]"
    prompt = prompt_format.generate_prompt(
        Prompt(
            prompt="hello1",
            use_prompt_format=False,
        )
    )
    assert prompt == "hello1"


def test_prompt_format():
    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="",
    )
    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[user] hello1 [/user] [assistant]"

    # User+system
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system] hello1 [/system] [user] hello2 [/user] [assistant]"

    # User+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[user] hello1 [/user] [assistant] hello2 [/assistant] [user] hello3 [/user] [assistant]"
    )

    # system+User+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] hello1 [/system] [user] hello2 [/user] [assistant] hello3 [/assistant] [user] hello4 [/user] [assistant]"
    )

    # User+assistant+user+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
        Message(role="assistant", content="hello4"),
        Message(role="user", content="hello5"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[user] hello1 [/user] [assistant] hello2 [/assistant] [user] hello3 [/user] [assistant] hello4 [/assistant] [user] hello5 [/user] [assistant]"
    )

    # system+User+assistant+user+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
        Message(role="assistant", content="hello5"),
        Message(role="user", content="hello6"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] hello1 [/system] [user] hello2 [/user] [assistant] hello3 [/assistant] [user] hello4 [/user] [assistant] hello5 [/assistant] [user] hello6 [/user] [assistant]"
    )

    # system+system+user
    # should error
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="hello"),
        Message(role="system", content="hello"),
    ]
    with pytest.raises(ValueError):
        prompt = prompt_format.generate_prompt(messages)

    # user+system+assistant
    # system should be moved to top
    messages = [
        Message(role="user", content="hello1"),
        Message(role="system", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system] hello2 [/system] [user] hello1 [/user] [assistant]"


def test_prompt_format_default_system_message():
    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="Test",
    )
    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system] Test [/system] [user] hello1 [/user] [assistant]"

    # User+system
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system] hello1 [/system] [user] hello2 [/user] [assistant]"

    # User+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] Test [/system] [user] hello1 [/user] [assistant] hello2 [/assistant] [user] hello3 [/user] [assistant]"
    )

    # system+User+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] hello1 [/system] [user] hello2 [/user] [assistant] hello3 [/assistant] [user] hello4 [/user] [assistant]"
    )

    # User+assistant+user+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
        Message(role="assistant", content="hello4"),
        Message(role="user", content="hello5"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] Test [/system] [user] hello1 [/user] [assistant] hello2 [/assistant] [user] hello3 [/user] [assistant] hello4 [/assistant] [user] hello5 [/user] [assistant]"
    )

    # system+User+assistant+user+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
        Message(role="assistant", content="hello5"),
        Message(role="user", content="hello6"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[system] hello1 [/system] [user] hello2 [/user] [assistant] hello3 [/assistant] [user] hello4 [/user] [assistant] hello5 [/assistant] [user] hello6 [/user] [assistant]"
    )

    # system+system+user
    # should error
    messages = [
        Message(role="system", content="hello"),
        Message(role="user", content="hello"),
        Message(role="system", content="hello"),
    ]
    with pytest.raises(ValueError):
        prompt = prompt_format.generate_prompt(messages)

    # user+system+assistant
    # system should be moved to top
    messages = [
        Message(role="user", content="hello1"),
        Message(role="system", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system] hello2 [/system] [user] hello1 [/user] [assistant]"


def test_prompt_format_system_in_user():
    with pytest.raises(ValidationError):
        # Should raise if system_in_user=True and
        # user doesn't have '{system}'
        prompt_format = PromptFormat(
            system="[system] {instruction} [/system] ",
            assistant="[assistant] {instruction} [/assistant] ",
            trailing_assistant="[assistant]",
            user="[user] {instruction} [/user] ",
            default_system_message="",
            system_in_user=True,
        )

    prompt_format = PromptFormat(
        system="<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        assistant=" {instruction} </s><s> ",
        trailing_assistant=" ",
        user="[INST] {system}{instruction} [/INST]",
        default_system_message="",
        system_in_user=True,
    )

    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] hello1 [/INST] "

    # User+system
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] "

    # User+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] hello1 [/INST] hello2 </s><s> [INST] hello3 [/INST] "

    # system+User+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] hello3 </s><s> [INST] hello4 [/INST] "
    )

    # User+assistant+user+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
        Message(role="assistant", content="hello4"),
        Message(role="user", content="hello5"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] hello1 [/INST] hello2 </s><s> [INST] hello3 [/INST] hello4 </s><s> [INST] hello5 [/INST] "
    )

    # system+User+assistant+user+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
        Message(role="assistant", content="hello5"),
        Message(role="user", content="hello6"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] hello3 </s><s> [INST] hello4 [/INST] hello5 </s><s> [INST] hello6 [/INST] "
    )

    # user+system+assistant
    # system should be moved to top
    messages = [
        Message(role="user", content="hello1"),
        Message(role="system", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\nhello2\n<</SYS>>\n\nhello1 [/INST] "


def test_prompt_format_system_in_user_default_system_message():
    with pytest.raises(ValidationError):
        # Should raise if system_in_user=True and
        # user doesn't have '{system}'
        prompt_format = PromptFormat(
            system="[system] {instruction} [/system] ",
            assistant="[assistant] {instruction} [/assistant] ",
            trailing_assistant="[assistant]",
            user="[user] {instruction} [/user] ",
            default_system_message="",
            system_in_user=True,
        )

    prompt_format = PromptFormat(
        system="<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        assistant=" {instruction} </s><s> ",
        trailing_assistant=" ",
        user="[INST] {system}{instruction} [/INST]",
        default_system_message="Test",
        system_in_user=True,
    )

    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\nTest\n<</SYS>>\n\nhello1 [/INST] "

    # User+system
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] "

    # User+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nTest\n<</SYS>>\n\nhello1 [/INST] hello2 </s><s> [INST] hello3 [/INST] "
    )

    # system+User+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] hello3 </s><s> [INST] hello4 [/INST] "
    )

    # User+assistant+user+assistant+user
    messages = [
        Message(role="user", content="hello1"),
        Message(role="assistant", content="hello2"),
        Message(role="user", content="hello3"),
        Message(role="assistant", content="hello4"),
        Message(role="user", content="hello5"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nTest\n<</SYS>>\n\nhello1 [/INST] hello2 </s><s> [INST] hello3 [/INST] hello4 </s><s> [INST] hello5 [/INST] "
    )

    # system+User+assistant+user+assistant+user
    messages = [
        Message(role="system", content="hello1"),
        Message(role="user", content="hello2"),
        Message(role="assistant", content="hello3"),
        Message(role="user", content="hello4"),
        Message(role="assistant", content="hello5"),
        Message(role="user", content="hello6"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[INST] <<SYS>>\nhello1\n<</SYS>>\n\nhello2 [/INST] hello3 </s><s> [INST] hello4 [/INST] hello5 </s><s> [INST] hello6 [/INST] "
    )

    # user+system+assistant
    # system should be moved to top
    messages = [
        Message(role="user", content="hello1"),
        Message(role="system", content="hello2"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\nhello2\n<</SYS>>\n\nhello1 [/INST] "


def test_prompt_format_add_system_tags():
    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="",
        add_system_tags_even_if_message_is_empty=True,
    )

    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[system]  [/system] [user] hello1 [/user] [assistant]"

    prompt_format = PromptFormat(
        system="<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        assistant=" {instruction} </s><s> ",
        trailing_assistant=" ",
        user="[INST] {system}{instruction} [/INST]",
        default_system_message="",
        system_in_user=True,
        add_system_tags_even_if_message_is_empty=True,
    )

    # Only user, no system
    messages = [Message(role="user", content="hello1")]
    prompt = prompt_format.generate_prompt(messages)
    assert prompt == "[INST] <<SYS>>\n\n<</SYS>>\n\nhello1 [/INST] "


def test_prompt_format_strip_whitespace():
    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="",
        strip_whitespace=True,
    )

    # Only user, no system
    messages = [
        Message(role="user", content="hello1 "),
        Message(role="assistant", content=" hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[user] hello1 [/user] [assistant] hello2 [/assistant] [user] hello3 [/user] [assistant]"
    )

    prompt_format = PromptFormat(
        system="[system] {instruction} [/system] ",
        assistant="[assistant] {instruction} [/assistant] ",
        trailing_assistant="[assistant]",
        user="[user] {instruction} [/user] ",
        default_system_message="",
        strip_whitespace=False,
    )

    # Only user, no system
    messages = [
        Message(role="user", content="hello1 "),
        Message(role="assistant", content=" hello2"),
        Message(role="user", content="hello3"),
    ]
    prompt = prompt_format.generate_prompt(messages)
    assert (
        prompt
        == "[user] hello1  [/user] [assistant]  hello2 [/assistant] [user] hello3 [/user] [assistant]"
    )


def test_prompt_format_equivalency_llama():
    model = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompt_format = PromptFormat(
        system="<<SYS>>\n{instruction}\n<</SYS>>\n\n",
        assistant=" {instruction} </s><s>",
        trailing_assistant="",
        user="[INST] {system}{instruction} [/INST]",
        default_system_message="",
        system_in_user=True,
    )

    conversations = [
        [Message(role="user", content="hello1")],
        [
            Message(role="system", content="hello1"),
            Message(role="user", content="hello2"),
        ],
        [
            Message(role="user", content="hello1"),
            Message(role="assistant", content="hello2"),
            Message(role="user", content="hello3"),
        ],
        [
            Message(role="system", content="hello1"),
            Message(role="user", content="hello2"),
            Message(role="assistant", content="hello3"),
            Message(role="user", content="hello4"),
        ],
        [
            Message(role="user", content="hello1"),
            Message(role="assistant", content="hello2"),
            Message(role="user", content="hello3"),
            Message(role="assistant", content="hello4"),
            Message(role="user", content="hello5"),
        ],
        [
            Message(role="system", content="hello1"),
            Message(role="user", content="hello2"),
            Message(role="assistant", content="hello3"),
            Message(role="user", content="hello4"),
            Message(role="assistant", content="hello5"),
            Message(role="user", content="hello6"),
        ],
    ]
    for conversation in conversations:
        dict_conversation = [message.dict() for message in conversation]
        reference_tokens = tokenizer.apply_chat_template(
            dict_conversation, tokenize=True
        )
        our_tokens = tokenizer.encode(prompt_format.generate_prompt(conversation))
        assert reference_tokens == our_tokens


def test_prompt_format_equivalency_mistral():
    model = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model)

    prompt_format = PromptFormat(
        system="{instruction} + ",
        assistant="{instruction}</s> ",
        trailing_assistant="",
        user="[INST] {system}{instruction} [/INST]",
        default_system_message="",
        system_in_user=True,
    )

    conversations = [
        [Message(role="user", content="hello1")],
        [
            Message(role="user", content="hello1"),
            Message(role="assistant", content="hello2"),
            Message(role="user", content="hello3"),
        ],
        [
            Message(role="user", content="hello2"),
            Message(role="assistant", content="hello3"),
            Message(role="user", content="hello4"),
        ],
        [
            Message(role="user", content="hello1"),
            Message(role="assistant", content="hello2"),
            Message(role="user", content="hello3"),
            Message(role="assistant", content="hello4"),
            Message(role="user", content="hello5"),
        ],
    ]
    for conversation in conversations:
        dict_conversation = [message.dict() for message in conversation]
        reference_tokens = tokenizer.apply_chat_template(
            dict_conversation, tokenize=True
        )
        our_tokens = tokenizer.encode(prompt_format.generate_prompt(conversation))
        assert reference_tokens == our_tokens
