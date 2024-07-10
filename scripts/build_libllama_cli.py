import os
import shutil
import subprocess

def build():
    env = os.environ.copy()
    env['CXXFLAGS'] = env.get('CXXFLAGS', ' ') + ' -DSHARED_LIB'
    env['LDFLAGS'] = env.get('LDFLAGS', ' ') + ' -shared -o libllama-cli.so'
    
    if 'PYODIDE' in env and env['PYODIDE'] == '1':
        env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
        env['UNAME_M'] = 'wasm'

    subprocess.run(['rm', '-rf', 'llama.cpp'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_1.patch'], check=True)
    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli', 'GGML_NO_OPENMP=1', 'GGML_NO_LLAMAFILE=1'], check=True, env=env)

    shutil.copy('llama.cpp/libllama-cli.so', 'llama/libllama-cli.so')
