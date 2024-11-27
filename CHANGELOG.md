# CHANGELOG

## v0.1.23

Added:
  - Support and examples for `llava` and `minicpmv` models.

## v0.1.22

Added:
  - `llava` high-level API calls
  - `minicpmv` high-level API support

Changed:
    - Updated `llama.cpp`.

## v0.1.21

Changed:
    - Updated `llama.cpp`.

## v0.1.20

Changed:
    - Updated `llama.cpp`.

## v0.1.19

Changed:
    - Updated `llama.cpp` with **RWKV 6** support.

## v0.1.18

Changed:
    - Updated `llama.cpp` with **RWKV** support.

## v0.1.17

Added:
    - `LLAMA_CPP_BACKEND` which can be `cuda`, `vulkan` or `cpu`.

Changed:
    - Updated `llama.cpp`.
    - Updated requirements.
    - CUDA backend imports only 12.6 library.

Fixed:
    - `Options.top_p` check using `isinstance`.

## v0.1.16

Changed:
    - Updated `llama.cpp`.

## v0.1.15

Added:
    - `SmolLM-1.7B-Instruct-v0.2` examples.

Changed:
    - Updated `llama.cpp`.

## v0.1.14

Fixed:
    - Vulkan detection.

## v0.1.13

Fixed:
    - CUDA and Vulkan detection.

## v0.1.12

Added:
    - Build `vulkan_1_x` for general GPU.
    - Build `cuda 12.4.1` as default.

Changed:
    - Renamed examples for TinyLlama (chat, tool calling) and OpenAI.
    - Updated demo models definitions.
    - Updated examples (chat, tool calling).
    - `get_special_tokens` not supports parameter `force_standard_special_tokens: bool=False` which bypasses tokenizer's special tokens with standard/common ones.
    - Build `cuda 12.5.1` as additional build target but packaged on PyPI.
    - Build `cuda 12.6` as additional build target but packaged on PyPI.
    - Build `openblas` as additional build target but packaged on PyPI.

Fixed:
    - Handle `Options.no_display_prompt` on Python side.

## v0.1.11

Changed:
    - OpenAI: allow import of `routes` and `v1_chat_completions` handler.
    - `examples/demo_0.py`, tool calling example.

## v0.1.10

Added:
    - In `openai`, support for `prompt` and `extra_body`. Reference: https://github.com/openai/openai-python/blob/195c05a64d39c87b2dfdf1eca2d339597f1fce03/src/openai/resources/completions.py#L41
    - Pass `llama-cli` options to `openai`.
    - `util` module with `is_cuda_available` function.
    - `openai` supports both `prompt` and `messages`. Reference: https://github.com/openai/openai-python/blob/195c05a64d39c87b2dfdf1eca2d339597f1fce03/src/openai/resources/completions.py#L45

## v0.1.9

Added:
    - Support for default CPU tinyBLAS (llamafile, sgemm) builds.
    - Support for CPU OpenBLAS (GGML_OPENBLAS) builds.

Changed:
    - Build scripts now have separate step/function `cuda_12_5_1_setup` which setups CUDA 12.5.1 env for build-time.

Fixed:
    - Stop thread in `llama_generate` on `GeneratorExit`.

Removed:
    - `callback` parameter in `llama_generate` and dependent functions.

## v0.1.8

Added:
    - `Model.tokenizer_hf_repo` as optional in case when `Model.creator_hf_repo` cannot be used to tokenize / format prompt/messages.

## v0.1.7

Added:
    - Support for `stop` tokens/words.

Changed:
    - `llama/llama_cli.py` unified CPU and CUDA 12.5 modules into single module.

Removed:
    - Removed separate examples for CPU and CUDA 12.5 modules.

## v0.1.6

Changed:
    - Updated `huggingface-hub`.

Fixed:
    - `llama.__init__` now correctly imports submodules and handles CPU and CUDA backends.
    - OpenAI: `ctx_size: int = config.max_position_embeddings if max_tokens is None else max_tokens`.

## v0.1.5

Fixed:
    - Build for linux, upx uses best compression option, 7z uses more aggressive compression.
    - Do not use UPX for shared/dynamic library compression.

## v0.1.4

Added:
    - README: supported GPU Compute Capability for CUDA.

Fixed:
    - Cleaned up `build.py`.
    - Type annotations in OpenAI related code.

## v0.1.3

Added:
    - Support for PyPy 3.10 versions.

Changed:
    - Disabled GitHub Actions.
    - Uses `upx -9` to compress shared/dynamic libraries.
    - Repacks `whl` with better compression rate.
    - Auto-detect CUDA support.

Removed:
    - ctypes version and demos.

## v0.1.2

Added:
    - Preparation for [Chat Completions API by OpenAI ©](https://platform.openai.com/docs/overview) compatible server.

Fixed:
    - Argument `options` is `deepcopy`-ed when passed to `llama_generate(options)`, so it can be reused.

Changed:
    - Build for `manylinux_2_28` and `musllinux_1_2`.
    - Build for [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 6.1.

## v0.1.1

Changed:
    - Updated: `huggingface-hub = "^0.24.0"`, `setuptools = "^71.0.3"`

## v0.1.0

Added:
    - Park first version.
