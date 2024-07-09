import os
import shutil
import subprocess


def build_shared():
    env = os.environ.copy()
    env['CXXFLAGS'] = '-DSHARED_LIB'
    env['LDFLAGS'] = '-shared -o libllama-cli.so'

    subprocess.run(['rm', '-rf', 'llama.cpp'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_0.patch'], check=True)
    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli'], check=True, env=env)

    shutil.copy("llama.cpp/libllama-cli.so", "llama/libllama-cli.so", check=True)


def build_static():
    env = os.environ.copy()
    env['CXXFLAGS'] = '-DSHARED_LIB'
    
    subprocess.run(['rm', '-rf', 'llama.cpp'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_0.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'makefile_static_library_0.patch'], check=True)
    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli-a', 'GGML_NO_OPENMP=1', 'GGML_NO_LLAMAFILE=1'], check=True, env=env)
