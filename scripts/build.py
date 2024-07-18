import os
import glob
import shutil
import subprocess

from cffi import FFI

from clean import clean_llama, clean_llama_cpp, clean


def clone_llama_cpp():
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_3.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'Makefile_3.patch'], check=True)


def build_cpu(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()

    #
    # build llama.cpp
    #
    if 'PYODIDE' in env and env['PYODIDE'] == '1':
        env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
        env['UNAME_M'] = 'wasm'

    subprocess.run([
        'make',
        '-C',
        'llama.cpp',
        '-j',
        'llama-cli-shared',
        'llama-cli-static',
        'GGML_NO_OPENMP=1',
        'GGML_NO_LLAMAFILE=1',
        # 'GGML_OPENBLAS=1',
    ], check=True, env=env)

    subprocess.run(['mv', 'llama.cpp/llama-cli.so', 'llama/llama-cli-cpu.so'], check=True)
    
    #
    # cffi
    #
    ffibuilder = FFI()

    ffibuilder.cdef('''
        typedef void (*_llama_yield_token_t)(const char * token);
        typedef int (*_llama_should_stop_t)(void);
        int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
    ''')

    ffibuilder.set_source(
        '_llama_cli_cpu',
        '''
        #include <stdio.h>
        
        typedef void (*_llama_yield_token_t)(const char * token);
        typedef int (*_llama_should_stop_t)(void);
        int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
        ''',
        libraries=[
            'stdc++',
        ],
        extra_objects=['../llama.cpp/llama-cli.a'],
    )

    ffibuilder.compile(tmpdir='build', verbose=True)

    # ctypes
    for file in glob.glob('build/*.so') + glob.glob('llama.cpp/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll') + glob.glob('llama.cpp/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib') + glob.glob('llama.cpp/*.dylib'):
        shutil.move(file, 'llama/')


def build_cuda_12_5(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()

    #
    # cuda env
    #
    cuda_file = 'cuda_12.5.1_555.42.06_linux.run'
    cuda_url = f'https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/{cuda_file}'
    cuda_output_dir = os.path.abspath('./cuda-12.5.1')

    env['PATH'] = env['PATH'] + f':{cuda_output_dir}/dist/bin'
    env['CUDA_PATH'] = f'{cuda_output_dir}/dist'

    # download cuda file
    subprocess.run(['wget', '-N', cuda_url, '-P', cuda_output_dir], check=True)
    
    # extract cuda file
    cmd = ['chmod', '+x', f'{cuda_output_dir}/{cuda_file}']
    subprocess.run(cmd, check=True)
    
    cmd = [
        f'{cuda_output_dir}/{cuda_file}',
        '--tar',
        'mxvf',
        '--wildcards',
        './builds/cuda_cccl/*',
        './builds/cuda_cudart/*',
        './builds/cuda_nvcc/*',
        './builds/libcublas/*',
        '-C',
        cuda_output_dir,
    ]
    subprocess.run(cmd, cwd=cuda_output_dir, check=True)

    cmd = ['mkdir', '-p', f'{cuda_output_dir}/dist']
    subprocess.run(cmd, check=True)

    cmd = f'cp -r {cuda_output_dir}/builds/cuda_cccl/* {cuda_output_dir}/dist'
    subprocess.run(cmd, shell=True, check=True)

    cmd = f'cp -r {cuda_output_dir}/builds/cuda_cudart/* {cuda_output_dir}/dist'
    subprocess.run(cmd, shell=True, check=True)

    cmd = f'cp -r {cuda_output_dir}/builds/cuda_nvcc/* {cuda_output_dir}/dist'
    subprocess.run(cmd, shell=True, check=True)

    cmd = f'cp -r {cuda_output_dir}/builds/libcublas/* {cuda_output_dir}/dist'
    subprocess.run(cmd, shell=True, check=True)

    #
    # build llama.cpp
    #
    if 'PYODIDE' in env and env['PYODIDE'] == '1':
        env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
        env['UNAME_M'] = 'wasm'

    subprocess.run([
        'make',
        '-C',
        'llama.cpp',
        '-j',
        'llama-cli-static',
        'llama-cli-shared',
        'GGML_NO_OPENMP=1',
        'GGML_NO_LLAMAFILE=1',
        'GGML_CUDA=1',
    ], check=True, env=env)

    subprocess.run(['mv', 'llama.cpp/llama-cli.so', 'llama/llama-cli-cuda-12_5.so'], check=True)

    #
    # cffi
    #
    ffibuilder = FFI()

    ffibuilder.cdef('''
        typedef void (*_llama_yield_token_t)(const char * token);
        typedef int (*_llama_should_stop_t)(void);
        int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
    ''')

    ffibuilder.set_source(
        '_llama_cli_cuda_12_5',
        '''
        #include <stdio.h>
        
        typedef void (*_llama_yield_token_t)(const char * token);
        typedef int (*_llama_should_stop_t)(void);
        int _llama_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop, int stop_on_bos_eos_eot);
        ''',
        libraries=[
            'stdc++',
        ],
        extra_objects=['../llama.cpp/llama-cli.a'],
    )

    ffibuilder.compile(tmpdir='build', verbose=True)

    # ctypes
    for file in glob.glob('build/*.so') + glob.glob('llama.cpp/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll') + glob.glob('llama.cpp/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib') + glob.glob('llama.cpp/*.dylib'):
        shutil.move(file, 'llama/')


def build(*args, **kwargs):
    clean()
    clone_llama_cpp()

    # cpu
    clean_llama_cpp()
    build_cpu(*args, **kwargs)

    # cuda 12.5
    clean_llama_cpp()
    build_cuda_12_5(*args, **kwargs)


if __name__ == '__main__':
    build()
