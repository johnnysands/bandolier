# Bandolier

Bandolier is a library to make it easier to deal with OpenAI ChatGPT functions.

To expose functions to OpenAI GPT services, we need to tell OpenAI about the
functions.  Ideally we could simply inspect the functions and use that information
to communicate to OpenAI, but unfortunately all the information we need isn't
available.

Type annotations could get us part of the way there, but they don't solve the
whole problem, and they can't represent some of more complex data structures
it's possible to use with ChatGPT.

Bandolier uses decorators to add annotations to functions with the missing
information.  These annotations are also validated against the information
that we can get via introspection, helping your code and configuration
stay in sync.

Bandolier also provides some convenience functions for working with OpenAI's APIs.
It helps you manage message history and process requests for function calls automatically.

## Example

```python
from bandolier import Bandolier, annotate_arguments, annotate_description


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


# can also use docstring for description
def get_location():
    """Get the user's location."""
    return "San Francisco, CA"


def main():
    bandolier = Bandolier()
    bandolier.add_function(get_weather)
    bandolier.add_function(get_location)
    bandolier.add_system_message("You are a helpful assistant.")

    while True:
        user_input = input("You: ")
        bandolier.add_user_message(user_input)
        messages = bandolier.run()
        for messages in messages:
            if message.role == "function":
                continue
            # message can have either or both of content and function_call
            if message.content:
                print(f"{message.role}: {message.content}")
            if "function_call" in message:
                print(f"{message.role}: {message.function_call.name}")


        print(f"{message.role}: {message.content}")


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
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        temperature=0.0,
    )

    return response["choices"][0]

# the model should be specified to bandolier as well so it can do
# the correct token accounting when maintaining history.
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