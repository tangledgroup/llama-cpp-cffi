from openai import OpenAI
from llama.model import Model


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='llama-cpp-cffi',
)

model = Model(
    creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    hf_repo='TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF',
    hf_file='tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
)

messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
    {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
    {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
]

def demo_chat_completions():
    print('demo_chat_completions:')

    response = client.chat.completions.create(
        model=str(model),
        messages=messages,
        temperature=0.0,
        stop=['```\n'],

        # llama-cpp-cffi
        extra_body=dict(
            batch_size=512,
            n_gpu_layers=22,
            main_gpu=0,
            cont_batching=True,
            flash_attn=True,
        ),
    )

    print(response.choices[0].message.content)


def demo_chat_completions_stream():
    print('demo_chat_completions_stream:')

    response = client.chat.completions.create(
        model=str(model),
        messages=messages,
        temperature=0.0,
        stop=['```\n'],
        stream=True,

        # llama-cpp-cffi
        extra_body=dict(
            batch_size=512,
            n_gpu_layers=22,
            main_gpu=0,
            cont_batching=True,
            flash_attn=True,
        ),
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, flush=True, end='')

    print()


if __name__ == '__main__':
    demo_chat_completions()
    demo_chat_completions_stream()