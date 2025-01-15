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

**Python** 3.10+ binding for [llama.cpp](https://github.com/ggerganov/llama.cpp) using **cffi**. Supports **CPU**, **Vulkan 1.x** (AMD, Intel and Nvidia GPUs) and **CUDA 12.6** (Nvidia GPUs) runtimes, **x86_64** and **aarch64** platforms.

NOTE: Currently supported operating system is **Linux** (`manylinux_2_28` and `musllinux_1_2`), but we are working on both **Windows** and **macOS** versions.

## News

- **Jan 14 2025, v0.4.14**: Modular llama.cpp build using `cmake` build system. Deprecated `make` build system.
- **Jan 1 2025, v0.3.1**: OpenAI compatible API, **text** and **vision** models. Added support for **Qwen2-VL** models. Hot-swap of models on demand in server/API.
- **Dec 9 2024, v0.2.0**: Low-level and high-level APIs: llama, llava, clip and ggml API.
- **Nov 27 2024, v0.1.22**: Support for Multimodal models such as **llava** and **minicpmv**.

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

## LLM Example

```python
from llama import Model

#
# first define and load/init model
#
model = Model(
    creator_hf_repo='HuggingFaceTB/SmolLM2-1.7B-Instruct',
    hf_repo='bartowski/SmolLM2-1.7B-Instruct-GGUF',
    hf_file='SmolLM2-1.7B-Instruct-Q4_K_M.gguf',
)

model.init(n_ctx=8 * 1024, gpu_layers=99)

#
# messages
#
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '1 + 1 = ?'},
    {'role': 'assistant', 'content': '2'},
    {'role': 'user', 'content': 'Evaluate 1 + 2 in Python.'},
]

completions = model.completions(
    messages=messages,
    predict=1 * 1024,
    temp=0.7,
    top_p=0.8,
    top_k=100,
)

for chunk in completions:
    print(chunk, flush=True, end='')

#
# prompt
#
prompt='Evaluate 1 + 2 in Python. Result in Python is'

completions = model.completions(
    prompt=prompt,
    predict=1 * 1024,
    temp=0.7,
    top_p=0.8,
    top_k=100,
)

for chunk in completions:
    print(chunk, flush=True, end='')
```

### References
- `examples/llm.py`
- `examples/demo_text.py`

## VLM Example

```python
from llama import Model

#
# first define and load/init model
#
model = Model( # 1.87B
    creator_hf_repo='vikhyatk/moondream2',
    hf_repo='vikhyatk/moondream2',
    hf_file='moondream2-text-model-f16.gguf',
    mmproj_hf_file='moondream2-mmproj-f16.gguf',
)

model.init(n_ctx=8 * 1024, gpu_layers=99)

#
# prompt
#
prompt = 'Describe this image.'
image = 'examples/llama-1.png'

completions = model.completions(
    prompt=prompt,
    image=image,
    predict=1 * 1024,
)

for chunk in completions:
    print(chunk, flush=True, end='')
```

### References
- `examples/vlm.py`
- `examples/demo_llava.py`
- `examples/demo_minicpmv.py`
- `examples/demo_qwen2vl.py`

## API

### Server - llama-cpp-cffi + OpenAI API

Run server first:

```bash
python -m llama.server
# or
python -B -u -m gunicorn --bind '0.0.0.0:11434' --timeout 300 --workers 1 --worker-class aiohttp.GunicornWebWorker 'llama.server:build_app()'
```

### Client - llama-cpp-cffi API / curl

```bash
#
# llm
#
curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
-d '{
    "gpu_layers": 99,
    "prompt": "Evaluate 1 + 2 in Python."
}'

curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
-d '{
    "creator_hf_repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "hf_repo": "bartowski/SmolLM2-1.7B-Instruct-GGUF",
    "hf_file": "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "gpu_layers": 99,
    "prompt": "Evaluate 1 + 2 in Python."
}'

curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
-d '{
    "creator_hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
    "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "hf_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "gpu_layers": 99,
    "prompt": "Evaluate 1 + 2 in Python."
}'

curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
-d '{
    "creator_hf_repo": "Qwen/Qwen2.5-7B-Instruct",
    "hf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
    "hf_file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    "gpu_layers": 99,
    "prompt": "Evaluate 1 + 2 in Python."
}'

#
# vlm - example 1
#
image_path="examples/llama-1.jpg"
mime_type=$(file -b --mime-type "$image_path")
base64_data=$(base64 -w 0 "$image_path")

cat << EOF > /tmp/temp.json
{
    "creator_hf_repo": "Qwen/Qwen2-VL-2B-Instruct",
    "hf_repo": "bartowski/Qwen2-VL-2B-Instruct-GGUF",
    "hf_file": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
    "mmproj_hf_file": "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
    "gpu_layers": 99,
    "prompt": "Describe this image.",
    "image": "data:$mime_type;base64,$base64_data"
}
EOF

curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
--data-binary "@/tmp/temp.json"

#
# vlm - example 2
#
image_path="examples/llama-1.jpg"
mime_type=$(file -b --mime-type "$image_path")
base64_data=$(base64 -w 0 "$image_path")

cat << EOF > /tmp/temp.json
{
    "creator_hf_repo": "Qwen/Qwen2-VL-2B-Instruct",
    "hf_repo": "bartowski/Qwen2-VL-2B-Instruct-GGUF",
    "hf_file": "Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
    "mmproj_hf_file": "mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
    "gpu_layers": 99,
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:$mime_type;base64,$base64_data"}
            }
        ]}
    ]
}
EOF

curl -XPOST 'http://localhost:11434/api/1.0/completions' \
-H "Content-Type: application/json" \
--data-binary "@/tmp/temp.json"
```

### Client - OpenAI © compatible Chat Completions API

```bash
#
# text
#
curl -XPOST 'http://localhost:11434/v1/chat/completions' \
-H "Content-Type: application/json" \
-d '{
    "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct:bartowski/SmolLM2-1.7B-Instruct-GGUF:SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": "Evaluate 1 + 2 in Python."
        }
    ],
    "n_ctx": 8192,
    "gpu_layers": 99
}'

#
# image
#
image_path="examples/llama-1.jpg"
mime_type=$(file -b --mime-type "$image_path")
base64_data=$(base64 -w 0 "$image_path")

cat << EOF > /tmp/temp.json
{
    "model": "Qwen/Qwen2-VL-2B-Instruct:bartowski/Qwen2-VL-2B-Instruct-GGUF:Qwen2-VL-2B-Instruct-Q4_K_M.gguf:mmproj-Qwen2-VL-2B-Instruct-f16.gguf",
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:$mime_type;base64,$base64_data"}
            }
        ]}
    ],
    "n_ctx": 8192,
    "gpu_layers": 99
}
EOF

curl -XPOST 'http://localhost:11434/v1/chat/completions' \
-H "Content-Type: application/json" \
--data-binary "@/tmp/temp.json"

#
# Client Python API for OpenAI
#
python -B examples/demo_openai.py
```

### References
- `examples/demo_openai.py`
