"""Experiment with using OpenAI chat functions."""
from bandolier import Bandolier, completion


def get_weather(location):
    """
    Get the weather for a location.

    Parameters
    ----------
    location : string
        The city and state, e.g. San Francisco, CA.
    """
    return {"temperature": 72, "conditions": ["sunny", "windy"]}


# main
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
    },
]

bandolier = Bandolier()
bandolier.add(get_weather)

# TODO:
# add a routine to work from bandolier to openai
# so i can do:
# for message in bandolier.loop():


# get user message
while True:
    text = input("You: ")
    messages.append({"role": "user", "content": text})

    # generate completion
    response = completion(messages, bandolier.config())
    message = response.message
    messages.append(message)

    while response.finish_reason == "function_call":
        print("function: ", message.function_call.name, message.function_call.arguments)
        message = bandolier.call(
            message.function_call.name, message.function_call.arguments
        )
        messages.append(message)
        print("response: ", message["name"], message["content"])

        response = completion(messages, bandolier.config())
        message = response.message
        messages.append(message)

    if response.finish_reason != "stop":
        raise Exception(f"Unexpected finish reason: {response.finish_reason}")

    print("System: ", message.content)
