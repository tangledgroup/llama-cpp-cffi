import os
import glob
import shutil
import subprocess
from pprint import pprint

from cffi import FFI

from clean import clean_llama_cpp, clean


# if 'PYODIDE' in env and env['PYODIDE'] == '1':
#     env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
#     env['UNAME_M'] = 'wasm'


def clone_llama_cpp():
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'Makefile_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/llava/llava-cli.cpp', 'llava-cli_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/llava/minicpmv-cli.cpp', 'minicpmv-cli_5.patch'], check=True)


def cuda_12_6_3_setup(*args, **kwargs):
    #
    # cuda env
    #
    cuda_file = 'cuda_12.6.3_560.35.05_linux.run'
    cuda_url = f'https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/{cuda_file}'
    cuda_output_dir = os.path.abspath('./cuda-12.6.3')
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
    env['CXXFLAGS'] = '-O3 -DLLAMA_LIB'
    print('build_cpu:')
    pprint(env)

    for name in ['llama', 'llava', 'minicpmv']:
        #
        # build llama.cpp
        #
        subprocess.run([
            'make',
            '-C',
            'llama.cpp',
            '-j',
            f'{name}-cli-static',
            'GGML_NO_OPENMP=1',
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(f'''
            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
        ''')

        ffibuilder.set_source(
            f'_{name}_cli_cpu',
            f'''
            #include <stdio.h>

            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''',
            libraries=['stdc++'],
            extra_objects=[f'../llama.cpp/lib{name}_cli.a'],
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


def build_vulkan_1_x(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()
    env['CXXFLAGS'] = '-O3 -DLLAMA_LIB'
    print('build_vulkan_1_x:')
    pprint(env)

    for name in ['llama', 'llava', 'minicpmv']:
        #
        # build llama.cpp
        #
        subprocess.run([
            'make',
            '-C',
            'llama.cpp',
            '-j',
            f'{name}-cli-static',
            'GGML_NO_OPENMP=1',
            'GGML_VULKAN=1',
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(f'''
            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
        ''')

        ffibuilder.set_source(
            f'_{name}_cli_vulkan_1_x',
            f'''
            #include <stdio.h>

            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''',
            libraries=[
                'stdc++',
                'vulkan',
            ],
            extra_objects=[f'../llama.cpp/lib{name}_cli.a'],
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


def build_linux_cuda_12_6_3(*args, **kwargs):
    # build static and shared library
    env = os.environ.copy()
    CIBUILDWHEEL = int(os.environ.get('CIBUILDWHEEL', '0'))

    #
    # cuda env
    #
    cuda_output_dir = cuda_12_6_3_setup()

    env['PATH'] =  f'{cuda_output_dir}/dist/bin:{env["PATH"]}'
    env['CUDA_PATH'] = f'{cuda_output_dir}/dist'
    env['CC'] = 'gcc' if CIBUILDWHEEL else 'gcc-13'
    env['CXX'] = 'g++' if CIBUILDWHEEL else 'g++-13'
    env['NVCC_PREPEND_FLAGS'] = ' ' if CIBUILDWHEEL else '-ccbin /usr/bin/g++-13'
    env['CUDA_DOCKER_ARCH'] = 'compute_61'
    env['CXXFLAGS'] = '-O3 -DLLAMA_LIB'
    env['LD_LIBRARY_PATH'] = '/project/cuda-12.6.3/dist/lib64:/project/cuda-12.6.3/dist/targets/x86_64-linux/lib:/project/cuda-12.6.3/dist/lib64/stubs:$LD_LIBRARY_PATH'
    env['CUDA_HOME'] = '/project/cuda-12.6.3/dist'
    env['NVCCFLAGS'] = '\
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_90,code=sm_90'

    print('build_linux_cuda_12_6_3:')
    pprint(env)

    for name in ['llama', 'llava', 'minicpmv']:
        #
        # build llama.cpp
        #
        subprocess.run([
            'make',
            '-C',
            'llama.cpp',
            '-j',
            f'{name}-cli-static',
            'GGML_NO_OPENMP=1',
            'GGML_CUDA=1',
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(f'''
            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
        ''')

        ffibuilder.set_source(
            f'_{name}_cli_cuda_12_6_3',
            f'''
            #include <stdio.h>

            typedef void (*_llama_yield_token_t)(const char * token);
            typedef int (*_llama_should_stop_t)(void);
            int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
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
            extra_objects=[f'../llama.cpp/lib{name}_cli.a'],
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
    env = os.environ.copy()

    # clean, clone
    clean()
    clone_llama_cpp()

    # cpu
    if env.get('GGML_CPU', '1') != '0':
        clean_llama_cpp()
        build_cpu(*args, **kwargs)

    # vulkan 1.x
    if env.get('GGML_VULKAN', '1') != '0' and env.get('AUDITWHEEL_ARCH') in ('x86_64', None):
        clean_llama_cpp()
        build_vulkan_1_x(*args, **kwargs)

    # cuda 12.6.3
    if env.get('GGML_CUDA', '1') != '0':
        if env.get('AUDITWHEEL_POLICY') in ('manylinux2014', 'manylinux_2_28', None) and env.get('AUDITWHEEL_ARCH') in ('x86_64', None):
            clean_llama_cpp()
            build_linux_cuda_12_6_3(*args, **kwargs)


if __name__ == '__main__':
    build()
