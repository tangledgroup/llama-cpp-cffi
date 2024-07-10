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
# build
#

# x86_64
poetry run cibuildwheel --output-dir wheelhouse --platform linux --arch x86_64 .

# aarch64
docker run --rm --privileged linuxkit/binfmt:v0.8
poetry run cibuildwheel --output-dir wheelhouse --platform linux --arch aarch64 .

# pyodide, pyscript, wasm (NOTE: cannot be published to PyPI)
# poetry run cibuildwheel --output-dir wheelhouse --platform pyodide .

#
# publish
#
poetry publish --dist-dir wheelhouse

#
# run demos
#
python -B examples/demo_cffi.py
python -B examples/demo_ctypes.py
python -m http.server -d examples/demo_pyonide -b "0.0.0.0" 5000
```

```bash
make -j llama-cli-shared llama-cli-static GGML_NO_OPENMP=1 GGML_NO_LLAMAFILE=1
```