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
    #     creator_hf_repo='HuggingFaceTB/SmolLM-1.7B-Instruct',
    #     hf_repo='mradermacher/SmolLM-1.7B-Instruct-GGUF',
    #     hf_file='SmolLM-1.7B-Instruct.Q4_K_M.gguf',
    # )

    # model = Model(
    #     creator_hf_repo='TinyLlama/TinyLlama_v1.1',
    #     hf_repo='QuantFactory/TinyLlama_v1.1-GGUF',
    #     hf_file='TinyLlama_v1.1.Q4_K_M.gguf',
    # )

    model = Model(
        creator_hf_repo='squeeze-ai-lab/TinyAgent-1.1B',
        hf_repo='squeeze-ai-lab/TinyAgent-1.1B-GGUF',
        hf_file='TinyAgent-1.1B-Q4_K_M.gguf',
        tokenizer_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    # https://github.com/SqueezeAILab/TinyAgent/blob/0074615dd05ae632cc2321f63b4290682897334d/src/llm_compiler/planner.py#L44
    
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
        'Use special value `"$R"` for all unknown or missing arguments of a tool. '
        'Here are the available tools:\n'
        '```json\n'
        f'{tools_str}\n'
        '```\n'
        '\n'
        # 'Use special value `"$R"` for all values or results from previous tool calls. '
        # 'Use special value `"$R"` for all unknown or missing arguments. '
        # 'Do not try to evaluate or guess arguments. '
        # 'Do not evaluate special `"$R"` values, just pass them as value to tool calls. '
        # '\n'
        # 'Do not try to evaluate, just output tool calls without explanations.'
        # 'Do not try to evaluate or guess arguments. '
        # 'Use special value `"$R"` for all unknown or missing arguments. '
    )

    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': 'Divide by 15 with 3. Subtract 3.'},
        {'role': 'assistant', 'content': (
            '```json\n'
            '[\n'
            '  {"function": "div", "arguments": {"a": 15, "b": 3}},\n'
            '  {"function": "sub", "arguments": {"a": "$R", "b": 3}},\n'
            ']\n'
            '```'
        )},
        {'role': 'user', 'content': 'Multiply 3 with 5. Divide by 15. Add 3.'},
        {'role': 'assistant', 'content': (
            '```json\n'
            '[\n'
            '  {"function": "mul", "arguments": {"a": 3, "b": 5}},\n'
            '  {"function": "div", "arguments": {"a": "$R", "b": 15}},\n'
            '  {"function": "add", "arguments": {"a": "$R", "b": 3}},\n'
            ']\n'
            '```'
        )},
        {'role': 'user', 'content': 'Add 1 and 2. Then multiply it with 5. Result divide by 3.'},
    ]

    # pprint(messages)

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
    # demo_0()
    # demo_1()
    demo_2()
