import glob
import subprocess


def clean():
    files = glob.glob('llama/*.so')
    subprocess.run(['rm', '-fv'] + files, check=True)
