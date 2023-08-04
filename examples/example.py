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


def get_location():
    """
    Get the user's current location.
    """
    return "San Francisco, CA"


def main():
    bandolier = Bandolier()
    bandolier.add_function(get_weather)
    bandolier.add_function(get_location)

    bandolier.add_message(
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    )

    while True:
        user_input = input("You: ")
        bandolier.add_message({"role": "user", "content": user_input})
        message = bandolier.run()
        print("System: ", message["content"])


if __name__ == "__main__":
    main()
