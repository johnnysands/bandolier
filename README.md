# Bandolier

Bandolier is a library to make it easier to deal with OpenAI ChatGPT functions.

There are two key classes:

Bandolier tracks the information on your functions and helps you build the
prompts you need to expose the functions to the chat interface.

Conversation stores your message history.  Bandolier uses the Conversation to
build prompts with appropriate history. You can use a different Conversation
in each call to Bandolier to make it easier to manage multiple conversation
streams.

## Example

```python
from bandolier import Bandolier, Conversation, annotate_arguments, annotate_description


# First set up your functions
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
    # api call here...
    return {"temperature": 72, unit: unit, "conditions": ["sunny", "windy"]}


# can also use docstring for description
def get_location():
    """Get the user's location."""
    # api call here...
    return "San Francisco, CA"


def main():
    # Set up the bandolier function configuration
    bandolier = Bandolier()
    bandolier.add_function(get_weather)
    bandolier.add_function(get_location)

    # create a new conversational conversation
    conversation = Conversation()
    conversation.add_system_message("You are a helpful assistant.")

    while True:
        user_input = input("You: ")
        conversation.add_user_message(user_input)

        # the messages returned here are for UI purposes.  The conversation has already been
        # updated with new messages.
        messages = bandolier.run(conversation)
        for message in messages:
            # this message indicates function output, and you probably don't need to do anything with it.
            if message.role == "function":
                continue

            # messages from the server can have content and/or function_call both
            if message.content:
                print(f"{message.role}: {message.content}")
            if "function_call" in message:
                print(f"{message.role}: {message.function_call.name}")


if __name__ == "__main__":
    main()
```

If you want more control, there are two other ways to use the Library. You can specify
a different model version:

```python
bandolier = Bandolier(model="gpt-4")
```
Or you can specify your own completion function.


```python
def completion(model, messages, functions=None):
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    if functions:
        request["functions"] = functions

    response = openai.ChatCompletion.create(**request)
    return response["choices"][0]


bandolier = Bandolier(completion_fn=completion)

```

Or you can handle all the communication yourself, and only use Bandolier
as a way to annotate your functions.

```python
def date():
    """returns today's date"""
    return datetime.date.isoformat(datetime.datetime.now())

bandolier = Bandolier()
bandolier.add_function(date)

functions = Bandolier.get_function_metadata()

result = openai.ChatCompletion.create({
    "messages": ...
    "functions": functions,
    ...
})
```
