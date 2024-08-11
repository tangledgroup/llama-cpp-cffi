import json
from pprint import pprint
from transformers.utils import get_json_schema

from llama import llama_generate, get_config, Model, Options


def demo_0():
    model = Model(
        creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        hf_repo='second-state/TinyLlama-1.1B-Chat-v1.0-GGUF',
        hf_file='TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '</s>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()


def demo_1():
    model = Model(
        creator_hf_repo='TinyLlama/TinyLlama_v1.1',
        hf_repo='QuantFactory/TinyLlama_v1.1-GGUF',
        hf_file='TinyLlama_v1.1.Q4_K_M.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Evaluate: 1 + 1 = ?'},
        {'role': 'assistant', 'content': '2'},
        {'role': 'user', 'content': 'Evaluate: 1 + 2 = ?'},
    ]

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '</s>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()


def demo_2():
    # model = Model(
    #     creator_hf_repo='TinyLlama/TinyLlama_v1.1',
    #     hf_repo='QuantFactory/TinyLlama_v1.1-GGUF',
    #     hf_file='TinyLlama_v1.1.Q4_K_M.gguf',
    # )

    model = Model(
        creator_hf_repo='TinyLlama/TinyLlama_v1.1_math_code',
        hf_repo='mjschock/TinyLlama_v1.1_math_code-Q4_K_M-GGUF',
        hf_file='tinyllama_v1.1_math_code-q4_k_m.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)
    
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

    messages = [
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

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        temp=0,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>', '</s>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()

if __name__ == '__main__':
    demo_0()
    demo_1()
    demo_2()
