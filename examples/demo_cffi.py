# import os
# import sys
# sys.path.append(os.path.abspath('.'))

from llama.llama_cli_cffi import llama_generate, Model, Options

from demo_models import models


def demo_model(model: Model, messages: list[dict]):
    options = Options(
        ctx_size=2048,
        predict=-2,
        model=model,
        prompt=messages,
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    print()


if __name__ == '__main__':
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
    ]

    for model in models:
        print(f'{model = }')
        demo_model(model, messages)
        print('-' * 80)