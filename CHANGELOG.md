# CHANGELOG

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
