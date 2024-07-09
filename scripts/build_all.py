import subprocess


def build():
    subprocess.run(['poetry', 'run', 'build-clean'])
    subprocess.run(['poetry', 'run', 'build-libllama-cli-shared'])
    subprocess.run(['poetry', 'run', 'build-libllama-cli-static'])
    subprocess.run(['poetry', 'run', 'build-llama-cli-cffi-static'])
