"""Experiment with using OpenAI chat functions."""
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
        bandolier.add_message({"role": "user", "content": user_input})
        message = bandolier.run()
        print("System: ", message["content"])


if __name__ == "__main__":
    main()
