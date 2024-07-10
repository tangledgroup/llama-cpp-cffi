import os
import glob
import shutil
import subprocess

from cffi import FFI


ffibuilder = FFI()

ffibuilder.cdef('''
    void llama_set_stdout(FILE* f);
    void llama_set_stderr(FILE* f);
    void llama_set_fprintf(int (*func)(FILE*, const char* format, ...));
    void llama_set_fflush(int (*func)(FILE*));
    const char* llama_get_metadata_as_json(int argc, char ** argv);
    void llama_free_metadata_as_json(const char * c_output);
    int llama_cli_main(int argc, char ** argv);
''')

ffibuilder.set_source(
    '_llama_cli',
    '''
    #include <stdio.h>
    
    void llama_set_stdout(FILE* f);
    void llama_set_stderr(FILE* f);
    void llama_set_fprintf(int (*func)(FILE*, const char* format, ...));
    void llama_set_fflush(int (*func)(FILE*));
    const char* llama_get_metadata_as_json(int argc, char ** argv);
    void llama_free_metadata_as_json(const char * c_output);
    int llama_cli_main(int argc, char ** argv);
    ''',
    libraries=['stdc++'],
    extra_objects=['../llama.cpp/libllama-cli.a'],
)


def build():
    env = os.environ.copy()
    env['CXXFLAGS'] = '-DSHARED_LIB'

    if 'PYODIDE' in env and env['PYODIDE'] == '1':
        env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
        env['UNAME_M'] = 'wasm'
    
    subprocess.run(['rm', '-rf', 'llama.cpp'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_1.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'makefile_static_library_0.patch'], check=True)
    subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli-a', 'GGML_NO_OPENMP=1', 'GGML_NO_LLAMAFILE=1'], check=True, env=env)
    
    ffibuilder.compile(tmpdir='build', verbose=True)
    
    for file in glob.glob('build/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib'):
        shutil.move(file, 'llama/')
