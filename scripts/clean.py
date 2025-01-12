import os
import glob
import shutil
import subprocess


def remove_llama_cpp():
    # if not os.path.exists('llama.cpp'):
    #     return

    shutil.rmtree('llama.cpp', ignore_errors=True)


def clean_llama():
    files = glob.glob('llama/*.a') + glob.glob('llama/*.so') + glob.glob('llama/*.dylib') + glob.glob('llama/*.dll')

    # subprocess.run(['rm', '-fv'] + files, check=True)
    for n in files:
       os.unlink(n)



def clean_llama_cpp():
    # if not os.path.exists('./llama.cpp'):
    #     return

    if not os.path.exists('llama.cpp/build'):
        return

    shutil.rmtree('llama.cpp/build', ignore_errors=True)


def clean():
    clean_llama()
    clean_llama_cpp()
    shutil.rmtree('llama.cpp', ignore_errors=True)
    shutil.rmtree('build', ignore_errors=True)
    shutil.rmtree('dist', ignore_errors=True)
    shutil.rmtree('wheelhouse', ignore_errors=True)
