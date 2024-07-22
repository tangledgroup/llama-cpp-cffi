# llama-cpp-cffi

<!--
[![Build][build-image]]()
[![Status][status-image]][pypi-project-url]
[![Stable Version][stable-ver-image]][pypi-project-url]
[![Coverage][coverage-image]]()
[![Python][python-ver-image]][pypi-project-url]
[![License][mit-image]][mit-url]
-->
[![Downloads](https://img.shields.io/pypi/dm/llama-cpp-cffi)](https://pypistats.org/packages/llama-cpp-cffi)
[![Supported Versions](https://img.shields.io/pypi/pyversions/llama-cpp-cffi)](https://pypi.org/project/llama-cpp-cffi)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Python** binding for [llama.cpp](https://github.com/ggerganov/llama.cpp) using **cffi** and **ctypes**. Supports **CPU** and **CUDA 12.5** execution.

## Install

Basic library install:

```bash
pip install llama-cpp-cffi
```

In case you want [Chat Completions API by OpenAI ©](https://platform.openai.com/docs/overview) compatible API:

```bash
pip install llama-cpp-cffi[openai]
```

**IMPORTANT:** If you want to take advantage of **Nvidia** GPU acceleration, make sure that you have installed **CUDA 12.5**. If you don't have CUDA 12.5 installed follow instructions here: https://developer.nvidia.com/cuda-downloads

## Example

### Library Usage

`examples/demo_0.py`

```python
from llama import llama_generate, Model, Options
from llama import get_config

model = Model(
    creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hf_repo='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    hf_file='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
)

config = get_config(model.creator_hf_repo)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '1 + 1 = ?'},
    {'role': 'assistant', 'content': '2'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
]

options = Options(
    ctx_size=config.max_position_embeddings,
    predict=-2,
    model=model,
    prompt=messages,
)

for chunk in llama_generate(options):
    print(chunk, flush=True, end='')

# newline
print()
```

### OpenAI © compatible Chat Completions (TBD)

Run OpenAI compatible server:

```bash
python -B llama/openai.py
```

Run example `examples/demo_1.py` using OpenAI module:

```python
from openai import OpenAI
from llama import Model

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='llama-cpp-cffi',
)

model = Model(
    creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hf_repo='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    hf_file='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '1 + 1 = ?'},
    {'role': 'assistant', 'content': '2'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'}
]


def demo_chat_completions():
    print('demo_chat_completions:')

    response = client.chat.completions.create(
        model=str(model),
        messages=messages,
        temperature=0.0,
    )

    print(response.choices[0].message.content)


def demo_chat_completions_stream():
    print('demo_chat_completions_stream:')

    response = client.chat.completions.create(
        model=str(model),
        messages=messages,
        temperature=0.0,
        stream=True,
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


if __name__ == '__main__':
    demo_chat_completions()
    demo_chat_completions_stream()
```

## Demos

```bash
#
# run demos
#
python -B examples/demo_0.py
python -B examples/demo_cffi_cpu.py
python -B examples/demo_cffi_cuda_12_5.py
```
