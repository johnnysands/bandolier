"""Experiment with using OpenAI chat functions."""

from box import Box
import inspect
import openai
import json
import tiktoken


# openai helper
def completion(messages, functions=None):
    request = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.0,
    }
    if functions:
        request["functions"] = functions

    response = openai.ChatCompletion.create(**request)

    return response["choices"][0]


def annotate_description(description):
    def decorator(function):
        function.__doc__ = description
        return function

    return decorator


def annotate_arguments(properties):
    def decorator(function):
        function.__properties__ = properties
        return function

    return decorator


class Bandolier:
    def __init__(
        self, completion_fn=completion, model="gpt-3.5-turbo", max_tokens=3000
    ):
        self.functions = {}
        self.function_metadata = []
        self.messages = []
        self.completion_fn = completion_fn
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def add_function(self, function):
        name = function.__name__
        description = function.__doc__ if hasattr(function, "__doc__") else ""
        properties = (
            function.__properties__ if hasattr(function, "__properties__") else {}
        )

        # Get the list of arguments from the function signature
        signature = inspect.signature(function)
        function_args = set(signature.parameters.keys())

        properties_args = set(properties.keys())
        if function_args != properties_args:
            raise ValueError(f"Arguments for function {name} do not match the schema.")

        required = []
        for param_name, param in signature.parameters.items():
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        metadata = {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties},
            "required": required,
        }
        self.functions[name] = function
        self.function_metadata.append(metadata)

    def add_message(self, message):
        self.messages.append(Box(message))
        self._trim_messages()

    def add_system_message(self, content):
        "convience method for adding a system message"
        self.add_message({"role": "system", "content": content})

    def add_user_message(self, content):
        "convience method for adding a user message"
        self.add_message({"role": "user", "content": content})

    def call(self, function_name, arguments):
        arguments = json.loads(arguments)
        function = self.functions[function_name]
        return Box(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function(**arguments)),
            }
        )

    def get_function_metadata(self):
        return self.function_metadata

    def run(self):
        response = self.completion_fn(self.messages, self.get_function_metadata())
        message = response.message
        self.add_message(message)

        # TODO FIXME this code makes the assumption that function call will not have message content,
        # but that's not actually true.  It's possible to return a mesage with both content and a function call.
        # This needs to be refactored so that run can return multiple messages.
        messages = [message]
        while response.finish_reason == "function_call":
            message = self.call(
                message.function_call.name, message.function_call.arguments
            )
            self.add_message(message)
            messages.append(message)

            response = self.completion_fn(self.messages, self.get_function_metadata())
            message = response.message
            self.add_message(message)
            messages.append(message)

        if response.finish_reason != "stop":
            raise Exception(f"Unexpected finish reason: {response.finish_reason}")
        return messages

    def _trim_messages(self):
        # I'm not sure how accurate this is, but I encode each message to json and then count
        # the tokens in the result.

        token_count = sum(len(self.encoding.encode(m.to_json())) for m in self.messages)
        while token_count > self.max_tokens:
            removed_message = self.messages.pop(0)
            token_count -= len(self.encoding.encode(removed_message.to_json()))
