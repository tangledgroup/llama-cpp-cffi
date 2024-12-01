import glob
import subprocess


def clean_llama():
    files = glob.glob('llama/*.so') + glob.glob('llama/*.a') + glob.glob('llama/*.dylib') + glob.glob('llama/*.dll')
    subprocess.run(['rm', '-fv'] + files, check=True)


def clean_llama_cpp():
    subprocess.run([
        'make',
        '-C',
        'llama.cpp',
        'clean'
    ], check=True)


def clean():
    clean_llama()
    clean_llama_cpp()
    subprocess.run(['rm', '-fr', 'build'], check=True)
    subprocess.run(['rm', '-fr', 'dist'], check=True)
    subprocess.run(['rm', '-fr', 'llama.cpp'], check=True)
    subprocess.run(['rm', '-fr', 'wheelhouse'], check=True)
