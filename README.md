# llama-cpp-cffi

Python binding for llama.cpp using cffi

## Build

```bash
#
# setup venv
#
python -m venv venv
source venv/bin/activate
pip install poetry

#
# build step-by-step
#
# poetry run build-clean
# poetry run build-libllama-cli-shared
# poetry run build-libllama-cli-static
# poetry run build-llama-cli-cffi-static

#
# build
#
poetry run build-all


#
# run demos
#
python -B examples/demo_cffi.py
python -B examples/demo_ctypes.py
```
