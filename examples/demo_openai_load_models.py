from threading import Thread

from openai import OpenAI
from llama import Model, file_to_data_uri

from demo_models import demo_models


client = OpenAI(
    # base_url = 'http://localhost:11434/v1',
    base_url = 'http://openai.tangledlabs.com/v1',
    api_key='llama-cpp-cffi',
)


def demo_text_chat_completions_stream():
    print('demo_text_chat_completions_stream:')

    model_ids = [
        'arcee-ai/Llama-3.1-SuperNova-Lite',
        'Qwen/Qwen2.5-7B-Instruct',

        'Qwen/Qwen2.5-3B-Instruct',
        'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'Qwen/Qwen2.5-0.5B-Instruct',
        'HuggingFaceTB/SmolLM2-360M-Instruct',

        'microsoft/Phi-3.5-mini-instruct',
        'RWKV/v6-Finch-3B-HF',
        'RWKV/v6-Finch-1B6-HF',
    ]

    for model_id in model_ids:
        model: Model = demo_models[model_id]

        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
            {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
            {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
        ]

        response = client.chat.completions.create( # type: ignore
            model=str(model),
            messages=messages, # type: ignore
            temperature=0.0,
            stream=True,

            # llama-cpp-cffi
            extra_body=dict( # type: ignore
                n_ctx=4 * 1024,
                gpu_layers=5,
                predict=512,
            ),
        )

        print('model:', model)

        for chunk in response:
            print(chunk.choices[0].delta.content, flush=True, end='')

        print()
        print('='* 80)


def demo_text_chat_completions_stream_threads():
    print('demo_text_chat_completions_stream_threads:')

    model_ids = [
        'arcee-ai/Llama-3.1-SuperNova-Lite',
        'Qwen/Qwen2.5-7B-Instruct',

        'Qwen/Qwen2.5-3B-Instruct',
        'HuggingFaceTB/SmolLM2-1.7B-Instruct',
        'Qwen/Qwen2.5-1.5B-Instruct',
        'Qwen/Qwen2.5-0.5B-Instruct',
        'HuggingFaceTB/SmolLM2-360M-Instruct',

        'microsoft/Phi-3.5-mini-instruct',
        'RWKV/v6-Finch-3B-HF',
        'RWKV/v6-Finch-1B6-HF',
    ]

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

    def func(model):
        response = client.chat.completions.create( # type: ignore
            model=str(model),
            messages=messages, # type: ignore
            temperature=0.0,
            stream=True,

            # llama-cpp-cffi
            extra_body=dict( # type: ignore
                n_ctx=4 * 1024,
                gpu_layers=5,
                predict=512,
            ),
        )

        print('model:', model)

        for chunk in response:
            print(chunk.choices[0].delta.content, flush=True, end='')

        print()
        print('='* 80)

    threads = []

    for model_id in model_ids:
        model: Model = demo_models[model_id]
        t = Thread(target=func, args=(model,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    # demo_text_chat_completions_stream()
    demo_text_chat_completions_stream_threads()
