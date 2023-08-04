"""Experiment with using OpenAI chat functions."""

from box import Box
from docstring_parser import parse
import inspect
import openai
import json


# openai helper
def completion(messages, functions=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        temperature=0.0,
    )

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
    def __init__(self, completion_fn=completion):
        self.functions = {}
        self.function_metadata = []
        self.messages = []
        self.completion_fn = completion_fn

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

    def call(self, function_name, arguments):
        arguments = json.loads(arguments)
        function = self.functions[function_name]
        return {
            "role": "function",
            "name": function_name,
            "content": json.dumps(function(**arguments)),
        }

    def config(self):
        return self.function_metadata

    def run(self):
        response = self.completion_fn(self.messages, self.config())
        message = response.message
        self.add_message(message)

        while response.finish_reason == "function_call":
            message = self.call(
                message.function_call.name, message.function_call.arguments
            )
            self.add_message(message)

            response = self.completion_fn(self.messages, self.config())
            message = response.message
            self.add_message(message)

        if response.finish_reason != "stop":
            raise Exception(f"Unexpected finish reason: {response.finish_reason}")

        return message

    def _gather_function_metadata(self, function):
        doc = parse(function.__doc__)
        metadata = {
            "name": function.__name__,
            "description": doc.short_description,
            "parameters": {"type": "object", "properties": {}},
            "required": [],
        }

        for param in doc.params:
            param_type = param.type_name
            param_name = param.arg_name
            description = param.description
            metadata["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": description,
            }

        # determined which parameters are required by looking at the function signature
        sig = inspect.signature(function)
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                metadata["required"].append(param_name)

        return metadata
