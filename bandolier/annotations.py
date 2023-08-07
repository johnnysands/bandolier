def annotate_description(description):
    """Annotates a function with a description, so that it can be used by Bandolier as
    an OpenAI chat function.
    This one is optional, because Bandolier will also use a docstring if it exists.

    Example:
    @annotate_description("Get the weather for a location.")
    def get_weather(location, unit="F"):
        return {"temperature": 72, unit: unit, "conditions": ["sunny", "windy"]}
    """

    def decorator(function):
        function.__doc__ = description
        return function

    return decorator


def annotate_arguments(properties):
    """Annotates a function with information about the arguments it requires, so that
    it can be used by Bandolier as an OpenAI chat function.

    Example:
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
    def get_weather(location, unit="F"):
        return {"temperature": 72, unit: unit, "conditions": ["sunny", "windy"]}
    """

    def decorator(function):
        function.__properties__ = properties
        return function

    return decorator
