import subprocess


def clean():
    # subprocess.run(['rm', '-v', 'llama/*.so'], check=True, shell=True)
    subprocess.run('ls -l llama/*.so', check=True, shell=True)
    subprocess.run(["ls", "-l", "/dev/null"], check=True, shell=True)