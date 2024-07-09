import subprocess

import build_clean
import build_libllama_cli
import build_llama_cli_cffi

def build(*args, **kwargs):
    print(f'build {args = }, {kwargs = }')
    # subprocess.run(['poetry', 'run', 'build-clean'])
    # subprocess.run(['poetry', 'run', 'build-libllama-cli'])
    # subprocess.run(['poetry', 'run', 'build-llama-cli-cffi'])
    build_clean.clean()
    build_libllama_cli.build()
    build_llama_cli_cffi.build()


if __name__ == '__main__':
    build()