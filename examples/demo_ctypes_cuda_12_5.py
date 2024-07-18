# import os
# import sys
# sys.path.append(os.path.abspath('.'))

from llama.llama_cli_ctypes_cuda_12_5 import llama_generate, Model, Options
from llama.formatter import get_config

from demo_models import models


def demo_model(model: Model, messages: list[dict]):
    config = get_config(model.creator_hf_repo)
    
    options = Options(
        ctx_size=32 * 1024 if model.creator_hf_repo == 'microsoft/Phi-3-mini-128k-instruct' else config.max_position_embeddings,
        predict=-2,
        gpu_layers=19 if model.creator_hf_repo == 'microsoft/Phi-3-mini-128k-instruct' else 99,
        # log_disable=False,
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
        config = get_config(model.creator_hf_repo)
        print(f'{model = }, {config.max_position_embeddings = }')
        demo_model(model, messages)
        print('-' * 80)