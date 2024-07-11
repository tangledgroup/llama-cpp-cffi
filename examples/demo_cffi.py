# import os
# import sys
# sys.path.append(os.path.abspath('.'))

import psutil
from llama.llama_cli_cffi import llama_generate, Model, Options

from demo_models import models


def demo1():
    options = Options(
        no_display_prompt=True,
        threads=psutil.cpu_count(logical=False),
        ctx_size=8192,
        predict=512,
        flash_attn=True,
        cont_batching=True,
        simple_io=True,
        log_disable=True,
        hf_repo=models[0].hf_repo,
        hf_file=models[0].hf_file,
        prompt='<|system|>\nYou are a helpful assistant.<|end|><|user|>\nEvaluate 1 + 2.<|end|>\n<|assistant|>\n',
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    print()


def demo2():
    options = Options(
        no_display_prompt=True,
        threads=psutil.cpu_count(logical=False),
        ctx_size=2048,
        predict=-2,
        flash_attn=True,
        cont_batching=True,
        simple_io=True,
        log_disable=True,
        hf_repo=models[1].hf_repo,
        hf_file=models[1].hf_file,
        prompt='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nEvaluate 1 + 2.<|im_end|>\n<|im_start|>assistant\n',
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    print()


def demo3():
    options = Options(
        no_display_prompt=True,
        threads=psutil.cpu_count(logical=False),
        ctx_size=2048,
        predict=-2,
        flash_attn=True,
        cont_batching=True,
        simple_io=True,
        log_disable=True,
        hf_repo=models[2].hf_repo,
        hf_file=models[2].hf_file,
        prompt='<|system|>\nYou are a helpful assistant.<|end|><|user|>\nEvaluate 1 + 2.<|end|>\n<|assistant|>\n',
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    print()


if __name__ == '__main__':
    demo1()
    demo2()
    demo3()
