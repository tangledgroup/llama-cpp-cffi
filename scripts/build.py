import os
import glob
import shutil
import subprocess
from pprint import pprint

from cffi import FFI

from clean import clean_llama, clean_llama_cpp, clean


def clone_llama_cpp():
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_3.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'Makefile_3.patch'], check=True)


def cuda_12_5_1_setup(*args, **kwargs):
    #
    # cuda env
    #
    cuda_file = 'cuda_12.5.1_555.42.06_linux.run'
    cuda_url = f'https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/{cuda_file}'
    cuda_output_dir = os.path.abspath('./cuda-12.5.1')
    cuda_file_path = os.path.join(cuda_output_dir, cuda_file)

    # download cuda file
    if not os.path.exists(cuda_file_path):
        cmd = ['mkdir', '-p', f'{cuda_output_dir}']
        
        subprocess.run(cmd, check=True)
        subprocess.run(['curl', '-o', cuda_file_path, cuda_url], check=True)
    
    # extract cuda file
    cmd = ['chmod', '+x', f'{cuda_output_dir}/{cuda_file}']
    subprocess.run(cmd, check=True)
    
    cmd = [
        f'{cuda_output_dir}/{cuda_file}',
        '--tar',
        'mxf',
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

    return cuda_output_dir


def build_cpu(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()
    env['CXXFLAGS'] = '-O3'
    
    # if 'PYODIDE' in env and env['PYODIDE'] == '1':
    #     env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
    #     env['UNAME_M'] = 'wasm'

    pprint(env)

    #
    # build llama.cpp
    #
    subprocess.run([
        'make',
        '-C',
        'llama.cpp',
        '-j',
        'llama-cli-static',
        'GGML_NO_OPENMP=1',
        'GGML_NO_LLAMAFILE=1',
    ], check=True, env=env)

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
        libraries=['stdc++'],
        extra_objects=['../llama.cpp/llama_cli.a'],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3', '-flto'],
    )

    ffibuilder.compile(tmpdir='build', verbose=True)

    #
    # copy compiled modules
    #
    for file in glob.glob('build/*.so') + glob.glob('llama.cpp/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll') + glob.glob('llama.cpp/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib') + glob.glob('llama.cpp/*.dylib'):
        shutil.move(file, 'llama/')


def build_linux_cuda_12_5(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()

    # if 'PYODIDE' in env and env['PYODIDE'] == '1':
    #     env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
    #     env['UNAME_M'] = 'wasm'

    #
    # cuda env
    #
    # cuda_file = 'cuda_12.5.1_555.42.06_linux.run'
    # cuda_url = f'https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/{cuda_file}'
    # cuda_output_dir = os.path.abspath('./cuda-12.5.1')
    # cuda_file_path = os.path.join(cuda_output_dir, cuda_file)
    cuda_output_dir = cuda_12_5_1_setup()

    env['PATH'] =  f'{cuda_output_dir}/dist/bin:{env["PATH"]}'
    env['CUDA_PATH'] = f'{cuda_output_dir}/dist'
    env['CUDA_DOCKER_ARCH'] = 'compute_61'
    env['CXXFLAGS'] = '-O3'
    env['NVCCFLAGS'] = '\
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90'

    pprint(env)

    # # download cuda file
    # if not os.path.exists(cuda_file_path):
    #     cmd = ['mkdir', '-p', f'{cuda_output_dir}']
    # 
    #     subprocess.run(cmd, check=True)
    #     subprocess.run(['curl', '-o', cuda_file_path, cuda_url], check=True)
    
    # # extract cuda file
    # cmd = ['chmod', '+x', f'{cuda_output_dir}/{cuda_file}']
    # subprocess.run(cmd, check=True)
    #
    # cmd = [
    #     f'{cuda_output_dir}/{cuda_file}',
    #     '--tar',
    #     'mxf',
    #     '--wildcards',
    #     './builds/cuda_cccl/*',
    #     './builds/cuda_cudart/*',
    #     './builds/cuda_nvcc/*',
    #     './builds/libcublas/*',
    #     '-C',
    #     cuda_output_dir,
    # ]
    # subprocess.run(cmd, cwd=cuda_output_dir, check=True)
    #
    # cmd = ['mkdir', '-p', f'{cuda_output_dir}/dist']
    # subprocess.run(cmd, check=True)
    #
    # cmd = f'cp -r {cuda_output_dir}/builds/cuda_cccl/* {cuda_output_dir}/dist'
    # subprocess.run(cmd, shell=True, check=True)
    #
    # cmd = f'cp -r {cuda_output_dir}/builds/cuda_cudart/* {cuda_output_dir}/dist'
    # subprocess.run(cmd, shell=True, check=True)
    #
    # cmd = f'cp -r {cuda_output_dir}/builds/cuda_nvcc/* {cuda_output_dir}/dist'
    # subprocess.run(cmd, shell=True, check=True)
    #
    # cmd = f'cp -r {cuda_output_dir}/builds/libcublas/* {cuda_output_dir}/dist'
    # subprocess.run(cmd, shell=True, check=True)

    #
    # build llama.cpp
    #
    subprocess.run([
        'make',
        '-C',
        'llama.cpp',
        '-j',
        'llama-cli-static',
        'GGML_NO_OPENMP=1',
        'GGML_NO_LLAMAFILE=1',
        'GGML_CUDA=1',
    ], check=True, env=env)

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
            'cuda',
            'cublas',
            'culibos',
            'cudart',
            'cublasLt',
        ],
        library_dirs=[
            f'{cuda_output_dir}/dist/lib64',
            f'{cuda_output_dir}/dist/targets/x86_64-linux/lib',
            f'{cuda_output_dir}/dist/lib64/stubs',
        ],
        extra_objects=['../llama.cpp/llama_cli.a'],
        extra_compile_args=['-O3'],
        extra_link_args=['-O3', '-flto'],
    )

    ffibuilder.compile(tmpdir='build', verbose=True)

    #
    # copy compiled modules
    #
    for file in glob.glob('build/*.so') + glob.glob('llama.cpp/*.so'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dll') + glob.glob('llama.cpp/*.dll'):
        shutil.move(file, 'llama/')

    for file in glob.glob('build/*.dylib') + glob.glob('llama.cpp/*.dylib'):
        shutil.move(file, 'llama/')


def build(*args, **kwargs):
    # clean, clone
    clean()
    clone_llama_cpp()

    # cuda 12.5
    if os.environ.get('AUDITWHEEL_POLICY') in ('manylinux2014', 'manylinux_2_28', None) and os.environ.get('AUDITWHEEL_ARCH') in ('x86_64', None):
        clean_llama_cpp()
        build_linux_cuda_12_5(*args, **kwargs)

    # cpu
    clean_llama_cpp()
    build_cpu(*args, **kwargs)


if __name__ == '__main__':
    build()
