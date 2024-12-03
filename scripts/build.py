import os
import re
import glob
import shutil
import platform
import subprocess
from pprint import pprint
from tempfile import NamedTemporaryFile

from cffi import FFI

from clean import clean_llama_cpp, clean


REPLACE_CODE_ITEMS = {
    'extern': ' ',
    'static': ' ',
    '_Noreturn __attribute__((format(printf, 3, 4)))': '',
    'static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);': '',
    'int32_t op_params[64 / sizeof(int32_t)];': 'int32_t op_params[16];',
    'void ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);': '',
    'struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);': '',
    'int ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);': '',
    'struct clip_ctx * clip_model_load_cpu(const char * fname, int verbosity);': '',
}


# if 'PYODIDE' in env and env['PYODIDE'] == '1':
#     env['CXXFLAGS'] += ' -msimd128 -fno-rtti -DNDEBUG -flto=full -s INITIAL_MEMORY=2GB -s MAXIMUM_MEMORY=4GB -s ALLOW_MEMORY_GROWTH '
#     env['UNAME_M'] = 'wasm'


def preprocess_library_code(cc: str, cflags: list[str], include_dirs: list[str], files: list[str]) -> str:
    source: str = subprocess.check_output([
        cc,
        *cflags,
        *[f'-I{n}' for n in include_dirs],
        '-E',
        *files
    ], text=True)

    return source


def filter_library_code(source: str) -> str:
    temp_source: list[str] | str = []
    is_relevant_code = False

    for i, line in enumerate(source.splitlines()):
        if line.startswith('#'):
            line_items: list[str] = line.split(' ')
            filename: str = line_items[2]

            if filename.startswith('/') or filename.startswith('"/'):
                is_relevant_code = False
            else:
                is_relevant_code = True

            continue

        if is_relevant_code:
            temp_source.append(line)

    source = '\n'.join(temp_source)
    return source


def replace_code(source: str, items: dict[str, str]) -> str:
    temp_source: list[str] | str = []

    for i, line in enumerate(source.splitlines()):
        for k, v in items.items():
            if k in line:
                line = line.replace(k, v)

        temp_source.append(line)

    source = '\n'.join(temp_source)
    return source


def remove_attribute_code(source: str) -> str:
    temp_source: list[str] | str = []
    scope = 0

    for i, line in enumerate(source.splitlines()):
        if '__attribute__' in line:
            scope = 0

            for j in range(line.index('__attribute__') + len('__attribute__'), len(line)):
                if line[j] == '(':
                    scope += 1
                    continue
                elif line[j] == ')':
                    scope -= 1

                if scope == 0:
                    line = line[:line.index('__attribute__')] + line[j + 1:]
                    break

        temp_source.append(line)

    source = '\n'.join(temp_source)
    return source


def remove_duplicate_decls_and_defs(source: str) -> str:
    with NamedTemporaryFile(suffix='.h', mode='w', delete=False) as f:
        f.write(source)
        f.seek(0)

        p = subprocess.run([
            'clang-tidy',
            f.name,
            '-checks="clang-diagnostic-redefinition,clang-diagnostic-redeclared"',
            '--',
            '-std=c11',
            '-ferror-limit=0',
        ], text=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output: str = p.stdout
        print(f'{f.name=}')

    # print(f'{type(output)=} {output=}')
    print(output)

    re_defs_decls = [
        n
        for n in output.splitlines()
        if 'error' in n and ('redefinition' in n or 'redeclared' in n)
    ]
    print(f'{re_defs_decls=}')
    source_lines = source.splitlines()
    # raise 1

    for n in re_defs_decls:
        _, line, col, *_ = n.split(':')
        line = int(line)
        col = int(col)
        j = line - 1
        m = source_lines[j]
        m = m.strip()

        if (m.startswith('typedef') or m.startswith('struct') or m.startswith('union') or m.startswith('enum')) and m.endswith('{'):
            scope = 1
            m = '/* ' + m
            source_lines[j] = m

            for j in range(line, len(source_lines) + 1):
                m = source_lines[j]
                m = m.strip()

                if '{' in m:
                    scope += m.count('{')
                    continue
                elif '}' in m:
                    scope -= m.count('}')

                    if scope == 0:
                        m = m + ' */'
                        source_lines[j] = m
                        break

    source = '\n'.join(source_lines)
    return source


def add_prefix_to_function(func_signature: str, prefix: str) -> str:
    # Regular expression pattern to match C function declarations
    pattern = r"""
        (\b[\w\s\*\(\)\[\],]*?)         # Match the return type (words, spaces, pointers, arrays, commas)
        \s*                             # Optional whitespace before the function name
        (\b\w+\b)                       # Match the function name (capturing group for renaming)
        \s*                             # Optional whitespace after the function name
        (\([^)]*?)                      # Match the function parameters (open parentheses but not mandatory closed)
        (\s*__attribute__\s*\(\(.*?\)\))? # Optionally match __attribute__((...)) syntax
        \s*                             # Optional whitespace
        (?=$|;|\n|\))                   # Lookahead for end of line, semicolon, or closing parenthesis
    """

    # Replacement function to add prefix to the function name
    def replacer(match):
        return f"{match.group(1)} {prefix}{match.group(2)}{match.group(3)}{match.group(4) or ''}"

    # Substitute using the pattern and replacement function
    return re.sub(pattern, replacer, func_signature, flags=re.VERBOSE)


def get_func_declarations(source_code: str) -> list[str]:
    def remove_comments(code: str) -> str:
        # Remove multi-line comments
        code = re.sub(r'/\*[^*]*\*+(?:[^/*][^*]*\*+)*/', '', code)
        # Remove single-line comments
        code = re.sub(r'//[^\n]*', '', code)
        return code

    def find_matching_brace(code: str, start: int) -> int:
        """Find the matching closing brace considering nested blocks"""
        count = 1
        i = start

        while i < len(code) and count > 0:
            if code[i] == '{':
                count += 1
            elif code[i] == '}':
                count -= 1

            i += 1

        return i - 1 if count == 0 else -1

    def extract_declarations(code: str) -> list[tuple]:
        results = []
        i = 0

        while i < len(code):
            # Find potential function start
            match = re.search(r'''
                # Return type
                (?:(?:static|extern|inline|const|volatile|unsigned|signed|struct|enum|union|long|short)\s+)*
                [\w_]+                    # Base type
                (?:\s*\*\s*|\s+)         # Pointers or whitespace
                (?:const\s+)*            # Optional const after pointer
                # Function name
                ([\w_]+)                 # Capture function name
                \s*
                # Parameters
                \(
                ((?:[^()]*|\([^()]*\))*)  # Parameters allowing one level of nested parentheses
                \)
                \s*
                (?:{|;)                   # Either opening brace or semicolon
            ''', code[i:], re.VERBOSE)

            if not match:
                break

            start = i + match.start()
            end = i + match.end()

            # Get everything before the function name to extract return type
            func_start = code[i:].find(match.group(1), match.start())
            return_type = code[start:i + func_start].strip()

            # If we found an opening brace, find its matching closing brace
            if code[end-1] == '{':
                closing_brace = find_matching_brace(code, end)

                if closing_brace == -1:
                    break
                # Skip the entire function body
                i = closing_brace + 1
            else:
                i = end

            results.append((return_type, match.group(1), match.group(2)))

        return results

    # Remove comments and normalize whitespace
    source_code = remove_comments(source_code)

    # Extract declarations
    declarations = []

    for return_type, func_name, params in extract_declarations(source_code):
        # Clean up parameters
        params = re.sub(r'\s+', ' ', params.strip())
        # Create declaration
        declaration = f"{return_type} {func_name}({params});"
        declaration = re.sub(r'\s+', ' ', declaration)
        declarations.append(declaration)

    return declarations


def replace_inline_code(source: str) -> (str, str):
    temp_source: list[str] | str = []
    inline_source: list[str] | str = []
    is_inline_code = False
    is_inline_block = False
    scope = 0

    for i, line in enumerate(source.splitlines()):
        line = line.strip()

        if line.startswith('static inline'):
            is_inline_code = True
            is_inline_block = '{' in line
            scope = line.count('{')
            scope -= line.count('}')

            line = line[len('static inline '):]
            line = add_prefix_to_function(line, '_inline_')
            inline_source.append(line)
            continue

        if is_inline_code:
            inline_source.append(line)

            if not is_inline_block:
                is_inline_block = '{' in line

            scope += line.count('{')
            scope -= line.count('}')

            if is_inline_block and scope == 0:
                is_inline_code = False
                is_inline_block = False
                continue
        else:
            temp_source.append(line)

    source = '\n'.join(temp_source)
    inline_source = '\n'.join(inline_source)

    # function declarations for inlined definitions
    func_declarations: list[str] | str = get_func_declarations(inline_source)
    func_declarations = '\n'.join(func_declarations)

    source += '\n\n' + func_declarations
    return source


def cleanup_code(source: str) -> str:
    source = '\n'.join([line.rstrip() for line in source.splitlines() if line])
    return source


def clone_llama_cpp():
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['patch', 'llama.cpp/Makefile', 'Makefile_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/llava/clip.h', 'clip_h_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/llava/clip.cpp', 'clip_cpp_5.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/examples/llava/llava.cpp', 'llava_cpp_5.patch'], check=True)
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

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            '-DLLAMA_LIB',
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            # './llama.cpp/common',
        ],
        files=[
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)

    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)
    print(_source)

    # build
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
            'LLAMA_MAKEFILE=1',
            'GGML_NO_OPENMP=1',
            *([] if platform.machine() == 'aarch64' else ['GGML_NO_CPU_AARCH64=1']),
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(
            f'''
                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''' + _source,
            override=True,
        )

        ffibuilder.set_source(
            f'_{name}_cli_cpu',
            f'''
                #include <stdio.h>
                #include "llama.h"
                #include "llava/clip.h"
                #include "llava/llava.h"

                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''',
            libraries=[
                'stdc++',
                'm',
                'pthread',
            ],
            extra_objects=[],
            extra_compile_args=[
                '-O3',
                '-g',
                '-fPIC',
                '-DLLAMA_SHARED',
                '-DLLAMA_LIB',
                '-I../llama.cpp/ggml/include',
                '-I../llama.cpp/include',
                '-I../llama.cpp/examples',
            ],
            extra_link_args=[
                '-O3',
                '-g',
                '-flto',
                '-L../llama.cpp',
                f'-l{name}_cli',
            ],
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

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            '-DLLAMA_LIB',
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            # './llama.cpp/common',
        ],
        files=[
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)

    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)
    print(_source)

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
            'LLAMA_MAKEFILE=1',
            'GGML_NO_OPENMP=1',
            'GGML_VULKAN=1',
            *([] if platform.machine() == 'aarch64' else ['GGML_NO_CPU_AARCH64=1']),
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(
            f'''
                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''' + _source,
            override=True,
        )

        ffibuilder.set_source(
            f'_{name}_cli_vulkan_1_x',
            f'''
                #include <stdio.h>
                #include "llama.h"
                #include "llava/clip.h"
                #include "llava/llava.h"

                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''',
            libraries=[
                'stdc++',
                'm',
                'pthread',
                'vulkan',
            ],
            extra_objects=[],
            extra_compile_args=[
                '-O3',
                '-g',
                '-fPIC',
                '-DLLAMA_SHARED',
                '-DLLAMA_LIB',
                '-I../llama.cpp/ggml/include',
                '-I../llama.cpp/include',
                '-I../llama.cpp/examples',
            ],
            extra_link_args=[
                '-O3',
                '-g',
                '-flto',
                '-L../llama.cpp',
                f'-l{name}_cli',
            ],
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

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            '-DLLAMA_LIB',
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            # './llama.cpp/common',
        ],
        files=[
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)

    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)
    print(_source)

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
            'LLAMA_MAKEFILE=1',
            'GGML_NO_OPENMP=1',
            'GGML_CUDA=1',
            *([] if platform.machine() == 'aarch64' else ['GGML_NO_CPU_AARCH64=1']),
        ], check=True, env=env)

        #
        # cffi
        #
        ffibuilder = FFI()

        ffibuilder.cdef(
            f'''
                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''' + _source,
            override=True,
        )

        ffibuilder.set_source(
            f'_{name}_cli_cuda_12_6_3',
            f'''
                #include <stdio.h>
                #include "llama.h"
                #include "llava/clip.h"
                #include "llava/llava.h"

                typedef void (*_llama_yield_token_t)(const char * token);
                typedef int (*_llama_should_stop_t)(void);
                int _{name}_cli_main(int argc, char ** argv, _llama_yield_token_t _llama_yield_token, _llama_should_stop_t _llama_should_stop);
            ''',
            libraries=[
                'stdc++',
                'm',
                'pthread',
                'cuda',
                'cublas',
                'culibos',
                'cudart',
                'cublasLt'
            ],
            library_dirs=[
                f'{cuda_output_dir}/dist/lib64',
                f'{cuda_output_dir}/dist/targets/x86_64-linux/lib',
                f'{cuda_output_dir}/dist/lib64/stubs',
            ],
            extra_objects=[],
            extra_compile_args=[
                '-O3',
                '-g',
                '-fPIC',
                '-DLLAMA_SHARED',
                '-DLLAMA_LIB',
                '-I../llama.cpp/ggml/include',
                '-I../llama.cpp/include',
                '-I../llama.cpp/examples',
            ],
            extra_link_args=[
                '-O3',
                '-g',
                '-flto',
                '-L../llama.cpp',
                f'-l{name}_cli',
            ],
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

    # remove, clone
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
