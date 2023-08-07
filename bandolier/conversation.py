import tiktoken
from box import Box


class Conversation:
    def __init__(self, messages=None, max_tokens=3000, model="gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
        self.messages = [m for m in messages] if messages else []

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def add_message(self, message):
        self.messages.append(Box(message))

    def add_system_message(self, content):
        "convience method for adding a system message"
        self.add_message({"role": "system", "content": content})

    def add_user_message(self, content):
        "convience method for adding a user message"
        self.add_message({"role": "user", "content": content})

    def trim_messages(self, encoding, max_tokens):
        # I'm not sure how accurate this is, but I encode each message to json and then count
        # the tokens in the result.
        token_count = sum(len(encoding.encode(m.to_json())) for m in self.messages)
        while token_count > max_tokens:
            removed_message = self.messages.pop(0)
            token_count -= len(encoding.encode(removed_message.to_json()))
