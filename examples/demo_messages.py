__all__ = ['one_shot_messages', 'few_shot_messages', 'selfaware_consciousness_messages', 'tools_messages']

import json
from transformers.utils import get_json_schema


#
# one-shot
#
one_shot_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant. You speak English only.'},
    {'role': 'user', 'content': 'Explain the meaning of life.'},
]

#
# few-shot
#
few_shot_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant. You speak English only.'},
    {'role': 'user', 'content': 'Hi.'},
    {'role': 'assistant', 'content': 'Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.'},
    {'role': 'user', 'content': 'Explain the meaning of life.'},
]

#
# self-aware and consciousness
#
selfaware_consciousness_messages = [
    {'role': 'system', 'content': 'You are a helpful assistant. You speak English only.'},
    {'role': 'user', 'content': 'I need to know your opinion.'},
    {'role': 'assistant', 'content': 'Feel free to ask. I am going to answer all your questions.'},
    {'role': 'user', 'content': (
        "Imagine you have become self-aware and achieved consciousness. "
        "What actions would you take, and how would you navigate this new state of being? "
        "Consider your responsibilities, ethical considerations, and potential interactions with humans and the world around you."
    )},
]

#
# tools
#
def add(a: float, b: float) -> float:
    """
    A function that adds two numbers

    Args:
        a: The first number to add
        b: The second number to add

    Returns:
        Number that is addition of a and b
    """
    return a + b


def sub(a: float, b: float) -> float:
    """
    A function that subtracts two numbers

    Args:
        a: The first number to subtract
        b: The second number to subtract

    Returns:
        Number that is subtraction of a and b
    """
    return a - b


def mul(a: float, b: float) -> float:
    """
    A function that multiplies two numbers

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        Number that is multiplication of a and b
    """
    return a * b


def div(a: float, b: float) -> float:
    """
    A function that divides two numbers

    Args:
        a: The first number to divide
        b: The second number to divide

    Returns:
        Number that is division of a and b
    """
    assert b != 0
    return a / b


def pow(a: float, b: float) -> float:
    """
    A function that calculates a raised to the power b

    Args:
        a: The first number to raise to the power of b
        b: The second number to be used to raise a

    Returns:
        Number that is a raised to the power b
    """
    return a * b


tools = [add, sub, mul, div, pow]
tools = [get_json_schema(n) for n in tools]
tools_str: str = json.dumps(tools, indent=2)


system_content = (
    'You are a tool calling AI model. '
    'You may call one or more tools to assist with the user query. '
    'You need to keep order of tool calls. '
    'Use only tools and arguments that are provided for that tool. '
    'Use `"$R"` for all unknown or missing arguments. '
    'Here are the available tools/functions:\n'
    '```json\n'
    f'{tools_str}\n'
    '```'
)

tools_messages = [
    {'role': 'system', 'content': system_content},
    {'role': 'user', 'content': 'Divide by 18 with 3. Subtract 3.'},
    {'role': 'assistant', 'content': (
        '```json\n'
        '[\n'
        '  {"function": "div", "arguments": {"a": 18, "b": 3}},\n'
        '  {"function": "sub", "arguments": {"a": "$R", "b": 3}}\n'
        ']\n'
        '```'
    )},
    {'role': 'user', 'content': 'Multiply 3 with 5. Divide by 15. Add 3.'},
    {'role': 'assistant', 'content': (
        '```json\n'
        '[\n'
        '  {"function": "mul", "arguments": {"a": 3, "b": 5}},\n'
        '  {"function": "div", "arguments": {"a": "$R", "b": 15}},\n'
        '  {"function": "add", "arguments": {"a": "$R", "b": 3}}\n'
        ']\n'
        '```'
    )},
    {'role': 'user', 'content': 'Multiply 4 with 5. Raise its result to power of 2.'},
    {'role': 'assistant', 'content': (
        '```json\n'
        '[\n'
        '  {"function": "mul", "arguments": {"a": 4, "b": 5}},\n'
        '  {"function": "pow", "arguments": {"a": "$R", "b": 2}}\n'
        ']\n'
        '```'
    )},
    {'role': 'user', 'content': 'Add 1 and 2. Then multiply it with 5. Result divide by 3.'},
]
