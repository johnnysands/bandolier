"""Experiment with using OpenAI chat functions."""

from docstring_parser import parse
import inspect
import openai
import json


class Bandolier:
    def __init__(self):
        self.functions = {}
        self.function_metadata = []

    def add(self, function):
        metadata = self._gather_function_metadata(function)
        self.functions[metadata["name"]] = function
        self.function_metadata.append(metadata)

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
        # TODO implement this
        raise NotImplementedError

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


# openai routines
def completion(messages, functions=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.0,
    )

    return response["choices"][0]
