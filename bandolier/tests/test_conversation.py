from bandolier import Bandolier, Conversation


def test_add_message():
    conversation = Conversation()
    bandolier = Bandolier()
    message = {"role": "system", "content": "You are a helpful assistant."}
    conversation.add_message(message)
    assert message in conversation.messages


def test_message_trimming():
    bandolier = Bandolier(max_tokens=20)

    conversation = Conversation()
    conversation.add_user_message("Here is the first message.")
    # this would ordinarily happen on bandolier's next run()
    conversation.trim_messages(bandolier.encoding, bandolier.max_tokens)
    assert len(conversation.messages) == 1

    # Add a message that exceeds the token limit
    conversation.add_user_message("Here is the second message.")
    conversation.trim_messages(bandolier.encoding, bandolier.max_tokens)
    assert len(conversation.messages) == 1
    assert conversation.messages[0].content == "Here is the second message."


def test_message_trimming_with_multiple_messages():
    # Add multiple messages
    conversation = Conversation()
    for i in range(6):
        conversation.add_user_message("Hello")

    bandolier = Bandolier(max_tokens=50)
    conversation.trim_messages(bandolier.encoding, bandolier.max_tokens)

    assert len(conversation.messages) == 4  # determined empirically
