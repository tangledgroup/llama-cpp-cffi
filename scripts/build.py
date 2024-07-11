import os
import glob
import shutil
import subprocess

from cffi import FFI

from clean import clean

ffibuilder = FFI()

ffibuilder.cdef('''
    typedef void (*_llama_yield_token_t)(const char * token);
    typedef int (*_llama_should_stop_t)(void);
    int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
''')

ffibuilder.set_source(
    '_llama_cli',
    '''
    #include <stdio.h>
    
    typedef void (*_llama_yield_token_t)(const char * token);
    typedef int (*_llama_should_stop_t)(void);
    int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
    ''',
    libraries=['stdc++'],
    extra_objects=['../llama.cpp/llama-cli.a'],
)


def build(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()

    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_3.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'Makefile_3.patch'], check=True)

    if 'PYODIDE' in env and env['PYODIDE'] == '1':
        env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
        env['UNAME_M'] = 'wasm'

    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli-shared', 'llama-cli-static', 'GGML_NO_OPENMP=1', 'GGML_NO_LLAMAFILE=1'], check=True, env=env)
    
    # cffi
    ffibuilder.compile(tmpdir='build', verbose=True)

    # ctypes
    for file in glob.glob('build/*.so') + glob.glob('llama.cpp/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll') + glob.glob('llama.cpp/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib') + glob.glob('llama.cpp/*.dylib'):
        shutil.move(file, 'llama/')


if __name__ == '__main__':
    clean()
    build()
