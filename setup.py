from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import os
import shutil

class CustomBuildExtCommand(build_ext):
    def run(self):
        env = os.environ.copy()
        env['CXXFLAGS'] = '-DSHARED_LIB'
        env['LDFLAGS'] = '-shared -o libllama-cli.so'

        subprocess.run(['rm', '-rf', 'llama.cpp'])
        subprocess.run(['rm', '-rf', 'llama/libllama-cli.so'])
        subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp.git'])
        subprocess.run(['patch', 'llama.cpp/examples/main/main.cpp', 'main_shared_library_0.patch'])
        subprocess.run(['make', '-C', 'llama.cpp', '-j', 'llama-cli'], check=True, env=env)

        shutil.copy("llama.cpp/libllama-cli.so", "llama/libllama-cli.so")
        
        # Ensure the build directory exists and copy the shared library to the build directory
        build_lib_dir = os.path.join(self.build_lib, 'llama')
        os.makedirs(build_lib_dir, exist_ok=True)
        self.copy_file('llama/libllama-cli.so', os.path.join(build_lib_dir, 'libllama-cli.so'))

        super().run()

setup(
    name='llama-cpp-cffi',
    version='0.0.2',
    description='Python binding for llama.cpp using cffi',
    author='Marko Tasic',
    author_email='mtasic85@gmail.com',
    url='https://github.com/mtasic85/llama-cpp-cffi',
    packages=find_packages(),
    include_package_data=True,
    package_data={'llama': ['libllama-cli.so']},
    cmdclass={
        'build_ext': CustomBuildExtCommand,
    },
    install_requires=[
        'attrs>=23.2.0',
        'huggingface-hub>=0.23.4',
        'cffi>=1.16.0',
    ],
)
