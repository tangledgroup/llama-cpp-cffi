# llama-cpp-cffi

<!--
[![Build][build-image]]()
[![Status][status-image]][pypi-project-url]
[![Stable Version][stable-ver-image]][pypi-project-url]
[![Coverage][coverage-image]]()
[![Python][python-ver-image]][pypi-project-url]
[![License][mit-image]][mit-url]
-->
[![PyPI](https://img.shields.io/pypi/v/llama-cpp-cffi)](https://pypi.org/project/llama-cpp-cffi/)
[![Supported Versions](https://img.shields.io/pypi/pyversions/llama-cpp-cffi)](https://pypi.org/project/llama-cpp-cffi)
[![PyPI Downloads](https://img.shields.io/pypi/dm/llama-cpp-cffi)](https://pypistats.org/packages/llama-cpp-cffi)
[![Github Downloads](https://img.shields.io/github/downloads/tangledgroup/llama-cpp-cffi/total.svg?label=Github%20Downloads)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Python** binding for [llama.cpp](https://github.com/ggerganov/llama.cpp) using **cffi**. Supports **CPU**, **Vulkan 1.x** and **CUDA 12.6** runtimes, **x86_64** and **aarch64** platforms.

NOTE: Currently supported operating system is Linux (`manylinux_2_28` and `musllinux_1_2`), but we are working on both Windows and MacOS versions.

## News

**Nov 27 2024, v0.1.22**: Support for Multimodal models such as **llava** and **minicpmv**.

## Install

Basic library install:

```bash
pip install llama-cpp-cffi
```

In case you want [OpenAI © Chat Completions API](https://platform.openai.com/docs/overview) compatible API:

```bash
pip install llama-cpp-cffi[openai]
```

**IMPORTANT:** If you want to take advantage of **Nvidia** GPU acceleration, make sure that you have installed **CUDA 12**. If you don't have CUDA 12.X installed follow instructions here: https://developer.nvidia.com/cuda-downloads .

GPU Compute Capability: `compute_61`, `compute_70`, `compute_75`, `compute_80`, `compute_86`, `compute_89` covering from most of GPUs from **GeForce GTX 1050** to **NVIDIA H100**. [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

## Example

### Library Usage

References:
- `examples/demo_tinyllama_chat.py`
- `examples/demo_tinyllama_tool.py`

```python
from llama import completions, get_config, Model, Options


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

for chunk in completions(options):
    print(chunk, flush=True, end='')

# newline
print()
```

### OpenAI © compatible Chat Completions API - Server and Client

Run OpenAI compatible server:

```bash
python -m llama.openai
# or
python -B -u -m gunicorn --bind '0.0.0.0:11434' --timeout 300 --workers 1 --worker-class aiohttp.GunicornWebWorker 'llama.openai:build_app()'
```

Run OpenAI compatible client `examples/demo_openai_0.py`:

```bash
python -B examples/demo_openai_0.py
```

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
python -B examples/demo_smollm_chat.py
python -B examples/demo_smollm_tool.py
python -B examples/demo_rwkv_chat.py
python -B examples/demo_rwkv_tool.py
```
