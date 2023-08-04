from bandolier import Bandolier, annotate_arguments, annotate_description
from box import Box
import pytest
from types import SimpleNamespace


# Mock functions to be used for testing
@annotate_arguments(
    {
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA.",
        },
        "unit": {
            "type": "string",
            "description": "The unit to return the temperature in, e.g. F or C.",
            "default": "F",
        },
    }
)
@annotate_description("Get the weather for a location.")
def get_weather(location, unit="F"):
    return {"temperature": 72, unit: unit, "conditions": ["sunny", "windy"]}


def get_location():
    """Get the user's location."""
    return "San Francisco, CA"


# Test cases
def test_add_function():
    bandolier = Bandolier()
    bandolier.add_function(get_weather)
    assert "get_weather" in bandolier.functions


def test_add_message():
    bandolier = Bandolier()
    message = {"role": "system", "content": "You are a helpful assistant."}
    bandolier.add_message(message)
    assert message in bandolier.messages


def test_call():
    bandolier = Bandolier()
    bandolier.add_function(get_weather)
    response = bandolier.call(
        "get_weather", '{"location": "San Francisco, CA", "unit": "F"}'
    )
    assert response["name"] == "get_weather"
    assert "temperature" in response["content"]


def dict_to_simplenamespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_simplenamespace(v)
    return SimpleNamespace(**d)


# Test cases
def test_run():
    def mock_completion_fn(messages, config):
        if messages[-1]["role"] == "user":
            response = {
                "message": {
                    "role": "system",
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco, CA", "unit": "F"}',
                    },
                },
                "finish_reason": "function_call",
            }
        else:
            response = {
                "message": {
                    "role": "assistant",
                    "content": "The weather in San Francisco, CA is 72F and sunny.",
                },
                "finish_reason": "stop",
            }
        # we convert these to Box so they function more like the real thing
        return Box(response)

    bandolier = Bandolier(completion_fn=mock_completion_fn)
    bandolier.add_function(get_weather)
    bandolier.add_message({"role": "system", "content": "You are a helpful assistant."})
    bandolier.add_message(
        {"role": "user", "content": "What is the weather in San Francisco, CA?"}
    )
    response = bandolier.run()
    assert response.role == "assistant"
    assert "The weather in San Francisco, CA is 72F and sunny." in response["content"]


def test_run_only_message():
    def mock_completion_fn_only_message(messages, config):
        response = {
            "message": {
                "role": "assistant",
                "content": "Hello, how can I assist you today?",
            },
            "finish_reason": "stop",
        }
        return Box(response)

    bandolier = Bandolier(completion_fn=mock_completion_fn_only_message)
    bandolier.add_message({"role": "system", "content": "You are a helpful assistant."})
    bandolier.add_message({"role": "user", "content": "Hello assistant"})
    response = bandolier.run()
    assert response.role == "assistant"
    assert "Hello, how can I assist you today?" in response.content


def test_run_two_function_calls():
    def mock_completion_fn_two_function_calls(messages, config):
        if messages[-1].role == "user":
            response = {
                "message": {
                    "role": "system",
                    "function_call": {
                        "name": "get_location",
                        "arguments": "{}",
                    },
                },
                "finish_reason": "function_call",
            }
        elif (
            messages[-1].role == "system"
            and messages[-1].function_call.name == "get_location"
        ):
            response = {
                "message": {
                    "role": "system",
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco, CA", "unit": "F"}',
                    },
                },
                "finish_reason": "function_call",
            }
        else:
            response = {
                "message": {
                    "role": "assistant",
                    "content": "The weather in San Francisco, CA is 72F and sunny.",
                },
                "finish_reason": "stop",
            }
        return Box(response)

    bandolier = Bandolier(completion_fn=mock_completion_fn_two_function_calls)
    bandolier.add_function(get_location)
    bandolier.add_function(get_weather)
    bandolier.add_message({"role": "system", "content": "You are a helpful assistant."})
    bandolier.add_message({"role": "user", "content": "What's the weather like?"})
    response = bandolier.run()
    assert response.role == "assistant"
    assert "The weather in San Francisco, CA is 72F and sunny." in response.content


def test_add_function_too_many_annotated_arguments():
    @annotate_arguments({"arg1": {"type": "string"}, "arg2": {"type": "string"}})
    def function_with_one_arg(arg1):
        pass

    bandolier = Bandolier()

    # Test adding a function with too many annotated arguments
    with pytest.raises(ValueError):
        bandolier.add_function(function_with_one_arg)


def test_add_function_too_few_annotated_arguments():
    @annotate_arguments({"arg1": {"type": "string"}})
    def function_with_two_args(arg1, arg2):
        pass

    bandolier = Bandolier()

    # Test adding a function with too few annotated arguments
    with pytest.raises(ValueError):
        bandolier.add_function(function_with_two_args)
