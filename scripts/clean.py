import os
import glob
import shutil
import subprocess


def clean_llama():
    files = glob.glob('llama/*.a') + glob.glob('llama/*.so') + glob.glob('llama/*.dylib') + glob.glob('llama/*.dll')
    subprocess.run(['rm', '-fv'] + files, check=True)


def clean_llama_cpp():
    if not os.path.exists('./llama.cpp'):
        return

    if not os.path.exists('llama.cpp/build'):
        return

    shutil.rmtree('llama.cpp/build')


def clean():
    clean_llama()
    clean_llama_cpp()
    subprocess.run(['rm', '-fr', 'build'], check=True)
    subprocess.run(['rm', '-fr', 'dist'], check=True)
    subprocess.run(['rm', '-fr', 'llama.cpp'], check=True)
    subprocess.run(['rm', '-fr', 'wheelhouse'], check=True)
