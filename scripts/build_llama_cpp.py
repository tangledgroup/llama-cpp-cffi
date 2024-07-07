import os
import shutil
import subprocess

def build():
    env = os.environ.copy()
    env['CXXFLAGS'] = '-DSHARED_LIB'
    env['LDFLAGS'] = '-shared -o libllama-cli.so'

    subprocess.run(['rm', '-rf', 'llama.cpp'])
    subprocess.run(['rm', '-rf', 'llama/libllama-cli.so'])
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'])
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_0.patch'])
    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli'], check=True, env=env)

    shutil.copy("llama.cpp/libllama-cli.so", "llama/libllama-cli.so")
