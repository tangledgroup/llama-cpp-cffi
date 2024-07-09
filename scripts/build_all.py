import subprocess


def build():
    subprocess.run(['poetry', 'run', 'build-clean'])
    subprocess.run(['poetry', 'run', 'build-libllama-cli'])
    subprocess.run(['poetry', 'run', 'build-llama-cli-cffi'])
