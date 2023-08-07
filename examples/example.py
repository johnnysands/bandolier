"""Experiment with using OpenAI chat functions."""
from bandolier import Bandolier, Conversation, annotate_arguments, annotate_description


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

    conversation = Conversation()
    conversation.add_system_message("Hello, I am a helpful assistant.")

    print("Try asking about the weather.\n\n")

    while True:
        user_input = input("You: ")
        conversation.add_user_message(user_input)
        messages = bandolier.run(conversation)
        for message in messages:
            if message.role == "function":
                continue
            elif message.role == "assistant":
                if message.content:
                    print(f"{message.role}: {message.content}")
                if "function_call" in message:
                    print(f"{message.role}: {message.function_call.name}")
            else:
                raise ValueError(f"Unknown role: {message.role}")


if __name__ == "__main__":
    main()
