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

```bash
pip install llama-cpp-cffi
```

## Example

```python
from llama.llama_cli_cffi_cpu import llama_generate, Model, Options
# from llama.llama_cli_cffi_cuda_12_5 import llama_generate, Model, Options
# from llama.llama_cli_ctypes_cuda import llama_generate, Model, Options
# from llama.llama_cli_ctypes_cuda_12_5 import llama_generate, Model, Options

from llama.formatter import get_config

model = Model(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
)

config = get_config(model.creator_hf_repo)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
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

## Demos

```BASH
#
# run demos
#
python -B examples/demo_cffi_cpu.py
python -B examples/demo_cffi_cuda_12_5.py

python -B examples/demo_ctypes_cpu.py
python -B examples/demo_ctypes_cuda_12_5.py

# python -m http.server -d examples/demo_pyonide -b "0.0.0.0" 5000
```
