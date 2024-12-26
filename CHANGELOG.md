# CHANGELOG

## v0.3.0

Added:
  - llama-cpp-cffi server - compatible with llama.cpp cli options instead of OpenAI

Changed:
  - `llama.cpp` revision `9ba399dfa7f115effc63d48e6860a94c9faa31b2`
  - Refactored `Options` class into two separate classes: `ModelOptions`, `CompletionsOptions`

Removed:
  - Removed ambiguous `Options` class

## v0.2.7

Changed:
  - In `format_messages`, optional `options` argument
  - `llama.cpp` revision `081b29bd2a3d91e7772e3910ce223dd63b8d7d26`

## v0.2.6

Changed:
  - `llama.cpp` revision `5437d4aaf5132c879acda0bb67f2f8f71da4c9fe`

## v0.2.5

Fixed:
  - Replaced `tokenizer.decode(new_token_id)` with custom `_common_token_to_piece(context, new_token_id, True)`

## v0.2.4

Fixed:
  - `sampler_init` because `llama_sampler_init_penalties` in `llama.cpp` changed its behaviour

## v0.2.3

Changed:
  - `llama.cpp` revision `4f51968aca049080dc77e26603aa0681ea77fe45`
  - Build process now has global variable `LLAMA_CPP_GIT_REF`

Fixed:
  - Issue with Phi 3.5 based models, `tokenizer.decode([new_token_id], clean_up_tokenization_spaces=False)`

## v0.2.2

Added:
  - `Model.free`

Changed:
  - Fixed revision of `llama.cpp` for all wheels
  - `llama.cpp` revision `c27ac678dd393af0da9b8acf10266e760c8a0912`
  - disabled `llama_kv_cache_seq_cp` in `_decode_tokens`

## v0.2.1

Fixed:
  - Batch "decode" process. NOTE: Encode part is missing for encoder-decoder models.
  - Thread-safe calls to the most important functions of llama, llava, clip, ggml API.

Removed:
  - `mllama_completions` for low-level function for Mllama-based VLMs

## v0.2.0

Added:
  - New high-level Python API
  - Low-level C API calls from llama.h, llava.h, clip.h, ggml.h
  - `completions` for high-level function for LLMs / VLMs
  - `text_completions` for low-level function for LLMs
  - `clip_completions` for low-level function for CLIP-based VLMs
  - WIP: `mllama_completions` for low-level function for Mllama-based VLMs

Changed:
  - All examples

Removed:
  - `llama_generate` function
  - `llama_cpp_cli`
  - `llava_cpp_cli`
  - `minicpmv_cpp_cli`

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
