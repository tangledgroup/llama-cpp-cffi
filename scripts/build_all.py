import subprocess


def build(*args, **kwargs):
    print(f'build {args = }, {kwargs = }')
    subprocess.run(['poetry', 'run', 'build-clean'])
    subprocess.run(['poetry', 'run', 'build-libllama-cli'])
    subprocess.run(['poetry', 'run', 'build-llama-cli-cffi'])


if __name__ == '__main__':
    build()