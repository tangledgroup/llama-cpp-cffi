import glob
import subprocess


def clean():
    files = glob.glob('llama/*.so')
    subprocess.run(['rm', '-fv'] + files, check=True)
    subprocess.run(['rm', '-fr', 'build'], check=True)
    subprocess.run(['rm', '-fr', 'dist'], check=True)
    # subprocess.run(['rm', '-fr', 'llama.cpp'], check=True)
    subprocess.run(['rm', '-fr', 'wheelhouse'], check=True)
