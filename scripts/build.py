import os
import re
import glob
import shutil
import platform
import subprocess
from pprint import pprint
from tempfile import NamedTemporaryFile

# set the compiler programmatically in your Python code before importing CFFI
env = os.environ
CIBUILDWHEEL = int(os.environ.get('CIBUILDWHEEL', '0'))
AUDITWHEEL_ARCH = os.environ.get('AUDITWHEEL_ARCH', None)

CUDA_VERSION = os.environ.get('CUDA_VERSION', '12.8.0')
CUDA_FILE = os.environ.get('CUDA_FILE', 'cuda_12.8.0_570.86.10_linux.run')
CUDA_ARCHITECTURES = os.environ.get('CUDA_ARCHITECTURES', '50;61;70;75;80;86;89;90;100;101;120')

# CUDA_VERSION = os.environ.get('CUDA_VERSION', '12.6.3')
# CUDA_FILE = os.environ.get('CUDA_FILE', 'cuda_12.6.3_560.35.05_linux.run')
# CUDA_ARCHITECTURES = os.environ.get('CUDA_ARCHITECTURES', '50;61;70;75;80;86;89;90')
# # CUDA_ARCHITECTURES = os.environ.get('CUDA_ARCHITECTURES', '50;61;70;75;80;86;89;90;100;101;120')

# CUDA_VERSION = os.environ.get('CUDA_VERSION', '12.4.1')
# CUDA_FILE = os.environ.get('CUDA_FILE', 'cuda_12.4.1_550.54.15_linux.run')
# CUDA_ARCHITECTURES = os.environ.get('CUDA_ARCHITECTURES', '50;61;70;75;80;86;89;90')
# # CUDA_ARCHITECTURES = os.environ.get('CUDA_ARCHITECTURES', '50;61;70;75;80;86;89;90;100;101;120')

# if CIBUILDWHEEL and AUDITWHEEL_ARCH not in (None, 'aarch64'):
#     gcc_toolset_path = '/opt/rh/gcc-toolset-12'
#     env['DEVTOOLSET_ROOTPATH'] = '/opt/rh/gcc-toolset-12/root'
#     env['PATH'] = f'{gcc_toolset_path}/root/usr/bin:{env["PATH"]}'

# env['CC'] = shutil.which('gcc' if CIBUILDWHEEL else 'gcc-12') # type: ignore
# env['CXX'] = shutil.which('g++' if CIBUILDWHEEL else 'g++-12') # type: ignore
# env['LD'] = shutil.which('gcc' if CIBUILDWHEEL else 'gcc-12') # type: ignore
# env['CC'] = shutil.which('clang' if CIBUILDWHEEL else 'clang') # type: ignore
# env['CXX'] = shutil.which('clang++' if CIBUILDWHEEL else 'clang++') # type: ignore
# env['LD'] = shutil.which('clang' if CIBUILDWHEEL else 'clang') # type: ignore
# env['CC'] = shutil.which('gcc-12') # type: ignore
# env['CXX'] = shutil.which('g++-12') # type: ignore
# env['LD'] = shutil.which('gcc-12') # type: ignore
env['CC'] = shutil.which('gcc') # type: ignore
env['CXX'] = shutil.which('g++') # type: ignore
env['LD'] = shutil.which('gcc') # type: ignore
# env['CC'] = shutil.which('clang') # type: ignore
# env['CXX'] = shutil.which('clang++') # type: ignore
# env['LD'] = shutil.which('clang') # type: ignore

# env['GGML_CPU'] = '0'
# env['GGML_VULKAN'] = '0'
# env['GGML_CUDA'] = '0'

from cffi import FFI # type: ignore # noqa

from clean import remove_llama_cpp, clean # type: ignore # noqa


LLAMA_CPP_GIT_REF = 'cc473cac7cea1484c1f870231073b0bf0352c6f9'

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


def clone_llama_cpp():
    subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'], check=True)
    subprocess.run(['git', 'reset', '--hard', LLAMA_CPP_GIT_REF], cwd='llama.cpp', check=True)
    subprocess.run(['patch', 'llama.cpp/common/json-schema-to-grammar.cpp', 'json_schema_to_grammar_cpp_7.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/common/json-schema-to-grammar.h', 'json_schema_to_grammar_h_8.patch'], check=True)
    subprocess.run(['patch', 'llama.cpp/common/json.hpp', 'json_hpp_7.patch'], check=True)
    # subprocess.run(['patch', 'llama.cpp/examples/llava/clip.h', 'clip_h_9.patch'], check=True)
    # subprocess.run(['patch', 'llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c', 'ggml_cpu_c_6.patch'], check=True)


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
        # print(f'{f.name=}')

    # print(f'{type(output)=} {output=}')
    # print(output)

    re_defs_decls = [
        n
        for n in output.splitlines()
        if 'error' in n and ('redefinition' in n or 'redeclared' in n)
    ]
    # print(f'{re_defs_decls=}')
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
                (?:\s*\*\s*|\s+)          # Pointers or whitespace
                (?:const\s+)*             # Optional const after pointer
                # Function name
                ([\w_]+)                  # Capture function name
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


def replace_inline_code(source: str) -> str:
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


def replace_code(source: str, items: dict[str, str]) -> str:
    for k, v in items.items():
        source = source.replace(k, v)

    return source


def cuda_setup(*args, **kwargs):
    #
    # cuda env
    #
    cuda_file = CUDA_FILE
    cuda_url = f'https://developer.download.nvidia.com/compute/cuda/{CUDA_VERSION}/local_installers/{cuda_file}'
    cuda_output_dir = os.path.abspath(f'./cuda-{CUDA_VERSION}')
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
    # env['CC'] = 'clang' # if platform.machine() == 'aarch64' else env.get('CC', 'gcc')
    # env['CXX'] = 'clang++' # if platform.machine() == 'aarch64' else env.get('CC', 'g++')
    env['CFLAGS'] = '-O3 -fPIC'
    env['CXXFLAGS'] = '-O3 -fPIC -std=c++17'
    env['LDFLAGS'] = '-O3 -fPIC -std=c++17'
    print('build_cpu:')
    pprint(env)

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            './llama.cpp/common',
        ],
        files=[
            './llama.cpp/ggml/include/gguf.h',
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
            './llama.cpp/common/json-schema-to-grammar.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)

    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)
    # print(_source)

    #
    # build llama.cpp
    #
    subprocess.run([
        'cmake',
        '-B',
        'build',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DGGML_OPENMP=OFF',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        # *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a+dotprod'] if platform.machine() == 'aarch64' else []),
        *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a'] if platform.machine() == 'aarch64' else []),
    ], check=True, env=env, cwd='llama.cpp')

    targets = [
        'ggml',
        'ggml-base',
        'ggml-cpu',
        'common',
        'llama',
        'llava_static',
    ]

    for target in targets:
        subprocess.run([
            'cmake',
            '--build',
            'build',
            '--config',
            'Release',
            '-j',
            '--target',
            target,
        ], check=True, env=env, cwd='llama.cpp')

    #
    # cffi
    #
    ffibuilder = FFI()

    ffibuilder.cdef(
        _source + '''
            void *malloc(size_t size);
            void free(void *ptr);
            void *memcpy(void *to, const void *from, size_t num_bytes);

            extern "Python" void llama_cpp_cffi_ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data);
        ''',
        override=True,
    )

    ffibuilder.set_source(
        '_llama_cpp_cpu',
        '''
            #include "gguf.h"
            #include "llama.h"
            #include "llava/clip.h"
            #include "llava/llava.h"
            #include "json-schema-to-grammar.h"
        ''',
        libraries=[
            'stdc++',
            'c',
            'pthread',
            'm',
        ],
        extra_objects=[],
        extra_compile_args=[
            # *env['CFLAGS'].split(),
            '-I../llama.cpp/ggml/include',
            '-I../llama.cpp/include',
            '-I../llama.cpp/examples',
            '-I../llama.cpp/common',
        ],
        extra_link_args=[
            # *env['LDFLAGS'].split(),
            '-L../llama.cpp/build/src',
            '-L../llama.cpp/build/ggml/src',
            '-L../llama.cpp/build/common',
            '-L../llama.cpp/build/examples/llava',
            '-lggml',
            '-lggml-base',
            '-lggml-cpu',
            '-lcommon',
            '-lllama',
            '-lllava_static',
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
    # env['CC'] = 'clang' # if platform.machine() == 'aarch64' else env.get('CC', 'gcc')
    # env['CXX'] = 'clang++' # if platform.machine() == 'aarch64' else env.get('CC', 'g++')
    env['CFLAGS'] = '-O3 -fPIC'
    env['CXXFLAGS'] = '-O3 -fPIC -std=c++17'
    env['LDFLAGS'] = '-O3 -fPIC -std=c++17'
    print('build_vulkan_1_x:')
    pprint(env)

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            './llama.cpp/common',
        ],
        files=[
            './llama.cpp/ggml/include/gguf.h',
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
            './llama.cpp/common/json-schema-to-grammar.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)


    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)
    # print(_source)

    #
    # build llama.cpp
    #
    subprocess.run([
        'cmake',
        '-B',
        'build',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DGGML_OPENMP=OFF',
        '-DGGML_VULKAN=ON',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        # *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a+dotprod'] if platform.machine() == 'aarch64' else []),
        *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a'] if platform.machine() == 'aarch64' else []),
    ], check=True, env=env, cwd='llama.cpp')

    targets = [
        'ggml',
        'ggml-base',
        'ggml-cpu',
        'ggml-vulkan',
        'common',
        'llama',
        'llava_static',
    ]

    for target in targets:
        subprocess.run([
            'cmake',
            '--build',
            'build',
            '--config',
            'Release',
            '-j',
            '--target',
            target,
        ], check=True, env=env, cwd='llama.cpp')

    #
    # cffi
    #
    ffibuilder = FFI()

    ffibuilder.cdef(
        _source + '''
            void *malloc(size_t size);
            void free(void *ptr);
            void *memcpy(void *to, const void *from, size_t num_bytes);

            extern "Python" void llama_cpp_cffi_ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data);
        ''',
        override=True,
    )

    ffibuilder.set_source(
        '_llama_cpp_vulkan_1_x',
        '''
            #include "gguf.h"
            #include "llama.h"
            #include "llava/clip.h"
            #include "llava/llava.h"
            #include "json-schema-to-grammar.h"
        ''',
        libraries=[
            'stdc++',
            'c',
            'pthread',
            'm',
            'vulkan',
        ],
        extra_objects=[],
        extra_compile_args=[
            # *env['CFLAGS'].split(),
            '-I../llama.cpp/ggml/include',
            '-I../llama.cpp/include',
            '-I../llama.cpp/examples',
            '-I../llama.cpp/common',
        ],
        extra_link_args=[
            # *env['LDFLAGS'].split(),
            '-L../llama.cpp/build/ggml/src',
            '-L../llama.cpp/build/ggml/src/ggml-vulkan',
            '-L../llama.cpp/build/common',
            '-L../llama.cpp/build/src',
            '-L../llama.cpp/build/examples/llava',
            '-lggml',
            '-lggml-base',
            '-lggml-cpu',
            '-lggml-vulkan',
            '-lcommon',
            '-lllama',
            '-lllava_static',
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


def build_linux_cuda_12_x_y(*args, **kwargs):
    #
    # cuda env
    #
    # cuda_output_dir = cuda_12_6_3_setup()
    # cuda_output_dir = cuda_12_8_0_setup()
    cuda_output_dir = cuda_setup()

    # build static and shared library
    env = os.environ.copy()
    # env['CC'] = 'clang' # if platform.machine() == 'aarch64' else env.get('CC', 'gcc')
    # env['CXX'] = 'clang++' # if platform.machine() == 'aarch64' else env.get('CC', 'g++')
    env['CFLAGS'] = '-O3 -fPIC'
    env['CXXFLAGS'] = '-O3 -fPIC -std=c++17'
    env['LDFLAGS'] = '-O3 -fPIC -std=c++17'
    env['PATH'] = f'{cuda_output_dir}/dist/bin:{env["PATH"]}'
    env['CUDA_PATH'] = f'{cuda_output_dir}/dist'
    env['LD_LIBRARY_PATH'] = f'/project/cuda-{CUDA_VERSION}/dist/lib64:/project/cuda-{CUDA_VERSION}/dist/targets/x86_64-linux/lib:/project/cuda-{CUDA_VERSION}/dist/lib64/stubs:$LD_LIBRARY_PATH'
    env['CUDA_HOME'] = f'/project/cuda-{CUDA_VERSION}/dist'
    env['NVCC_PREPEND_FLAGS'] = f'-ccbin {env.get("CC", "gcc")} -Xcompiler -fPIC -Wno-deprecated-gpu-targets'
    # env['NVCC_CCBIN'] = env['CXX']
    # env['NVCC_PREPEND_FLAGS'] = '-Wno-deprecated-gpu-targets'
    print(f'build_linux_cuda_{CUDA_VERSION}:')
    pprint(env)

    # pre-process header code
    _source = preprocess_library_code(
        cc=env.get('CC', 'gcc'),
        cflags=[
            *(['-DGGML_USE_CPU_AARCH64'] if platform.machine() == 'aarch64' else []),
        ],
        include_dirs=[
            './llama.cpp/ggml/include',
            './llama.cpp/common',
        ],
        files=[
            './llama.cpp/ggml/include/gguf.h',
            './llama.cpp/examples/llava/clip.h',
            './llama.cpp/examples/llava/llava.h',
            './llama.cpp/include/llama.h',
            './llama.cpp/common/json-schema-to-grammar.h',
        ],
    )

    # filter relevant header code
    _source = filter_library_code(_source)

    # filter our attribute code
    _source = remove_attribute_code(_source)

    # remove duplicate decls/defs
    _source = remove_duplicate_decls_and_defs(_source)

    # filter our static inline code
    _source = replace_inline_code(_source)

    # strip empty lines
    _source = cleanup_code(_source)

    # patch of source
    _source = replace_code(_source, REPLACE_CODE_ITEMS)
    # print(_source)

    #
    # build llama.cpp
    #
    subprocess.run([
        'cmake',
        '-B',
        'build',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DGGML_OPENMP=OFF',
        '-DGGML_CUDA=ON',
        '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
        # '-DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1',
        # '-DGGML_CUDA_GRAPHS=ON',
        # '-DGGML_CUDA_FA_ALL_QUANTS=ON',
        '-DGGML_NATIVE=OFF',
        '-DGGML_CCACHE=OFF',
        # * a semicolon-separated list of integers, each optionally
        #   followed by '-real' or '-virtual'
        # * a special value: all, all-major, native
        f'-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}',
        f'CMAKE_CUDA_COMPILER={env["CUDA_HOME"]}/bin/nvcc',
        # f'CMAKE_CUDA_HOST_COMPILER={env["CXX"]}',
        # *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a+dotprod'] if platform.machine() == 'aarch64' else []),
        # *(['-DGGML_NATIVE=OFF', '-DGGML_CPU_ARM_ARCH=armv8-a'] if platform.machine() == 'aarch64' else []),
    ], check=True, env=env, cwd='llama.cpp')

    targets = [
        'ggml',
        'ggml-base',
        'ggml-cpu',
        'ggml-cuda',
        'common',
        'llama',
        'llava_static',
    ]

    for target in targets:
        subprocess.run([
            'cmake',
            '--build',
            'build',
            '--config',
            'Release',
            '-j',
            '--target',
            target,
        ], check=True, env=env, cwd='llama.cpp')

    #
    # cffi
    #
    ffibuilder = FFI()

    ffibuilder.cdef(
        _source + '''
            void *malloc(size_t size);
            void free(void *ptr);
            void *memcpy(void *to, const void *from, size_t num_bytes);

            extern "Python" void llama_cpp_cffi_ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data);
        ''',
        override=True,
    )

    ffibuilder.set_source(
        '_llama_cpp_cuda_12_6_3',
        '''
            #include "gguf.h"
            #include "llama.h"
            #include "llava/clip.h"
            #include "llava/llava.h"
            #include "json-schema-to-grammar.h"
        ''',
        libraries=[
            'stdc++',
            'c',
            'pthread',
            'm',
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
            # *env['CFLAGS'].split(),
            '-I../llama.cpp/ggml/include',
            '-I../llama.cpp/include',
            '-I../llama.cpp/examples',
            '-I../llama.cpp/common',
        ],
        extra_link_args=[
            # *env['LDFLAGS'].split(),
            '-L../llama.cpp/build/ggml/src',
            '-L../llama.cpp/build/ggml/src/ggml-cuda',
            '-L../llama.cpp/build/common',
            '-L../llama.cpp/build/src',
            '-L../llama.cpp/build/examples/llava',
            '-lggml',
            '-lggml-base',
            '-lggml-cpu',
            '-lggml-cuda',
            '-lcommon',
            '-lllama',
            '-lllava_static',
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

    # remove all previous builds
    clean()

    # cpu
    if env.get('GGML_CPU', '1') != '0':
        remove_llama_cpp()
        clone_llama_cpp()
        build_cpu(*args, **kwargs)

    # vulkan 1.x
    if env.get('GGML_VULKAN', '1') != '0' and env.get('AUDITWHEEL_ARCH') in ('x86_64', None):
        remove_llama_cpp()
        clone_llama_cpp()
        build_vulkan_1_x(*args, **kwargs)

    # cuda 12.x.y
    if env.get('GGML_CUDA', '1') != '0':
        if env.get('AUDITWHEEL_POLICY') in ('manylinux2014', 'manylinux_2_28', 'manylinux_2_34', None) and env.get('AUDITWHEEL_ARCH') in ('x86_64', None):
            remove_llama_cpp()
            clone_llama_cpp()
            build_linux_cuda_12_x_y(*args, **kwargs)


if __name__ == '__main__':
    build()
