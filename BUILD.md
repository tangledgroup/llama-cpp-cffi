# llama-cpp-cffi

## Build

```bash
# setup venv
python -m venv venv
source venv/bin/activate
pip install poetry

# local build
# poetry install --all-extras

# clean build
poetry run clean

# x86_64
poetry run cibuildwheel --output-dir wheelhouse --platform linux --arch x86_64 .

# aarch64
docker run --rm --privileged linuxkit/binfmt:v0.8
poetry run cibuildwheel --output-dir wheelhouse --platform linux --arch aarch64 .

# pyodide, pyscript, wasm (NOTE: cannot be published to PyPI)
# poetry run cibuildwheel --output-dir wheelhouse --platform pyodide .

# publish
poetry publish --dist-dir wheelhouse
```
