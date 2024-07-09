import glob
import shutil

from cffi import FFI


ffibuilder = FFI()

ffibuilder.cdef('''
    void llama_set_stdout(FILE* f);
    void llama_set_stderr(FILE* f);
    void llama_set_fprintf(int (*func)(FILE*, const char* format, ...));
    void llama_set_fflush(int (*func)(FILE*));
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
    int llama_cli_main(int argc, char ** argv);
    ''',
    libraries=['stdc++'],
    extra_objects=['../llama.cpp/libllama-cli.a'],
)


def build_static():
    subprocess.run(['rm', '-rf', 'llama.cpp'], check=True)
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    
    ffibuilder.compile(tmpdir="build", verbose=True)
    
    for file in glob.glob('build/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib'):
        shutil.move(file, 'llama/')
