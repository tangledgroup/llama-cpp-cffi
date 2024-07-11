__all__ = ['format_messages']

from copy import deepcopy

from transformers import AutoTokenizer


FALLBACK_MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] + '\n\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: '  + message['content'] }}"
            "{% if not loop.last %}"
                "{{ '\n\n' }}"
            "{% endif %}"
        "{% endif %}"
    "{% endfor %}"
    "{{ 'Assistant: ' }}"
)

CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

MINICHAT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ bos_token + message['content'] + eos_token }}"
        "{% elif message['role'] == 'user' %}"
            "{{ bos_token + '[|User|] ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ bos_token + '[|Assistant|] '  + message['content'] }}"
            "{% if not loop.last %}"
                "{{ eos_token }}"
            "{% endif %}"
        "{% endif %}"
    "{% endfor %}"
    "{{ '<s>[|Assistant|] ' }}"
)

GEMMA_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '<start_of_turn>model\n' + message['content'] }}"
            "{% if not loop.last %}"
                "{{ '<end_of_turn>\n' }}"
            "{% endif %}"
        "{% endif %}"
    "{% endfor %}"
    "{{ '<start_of_turn>model\n' }}"
)

MISTRALLITE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '<|prompter|>' + message['content'] + '</s>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '<|assistant|>' + message['content'] }}"
            "{% if not loop.last %}"
                "{{ '</s>\n' }}"
            "{% endif %}"
        "{% endif %}"
    "{% endfor %}"
    "{{ '<|assistant|>' }}"
)


def create_alternate_messages(model_id: str, messages: list[dict], convert_system_to_user: bool=False) -> list[dict]:
    assert messages
    messages = deepcopy(messages)

    if convert_system_to_user:
        for i, m in enumerate(list(messages)):
            if m['role'] == 'system':
                m['role'] = 'user'

    prev_m: dict = messages[0]
    new_messages: list[dict] = [prev_m]

    for i, m in enumerate(list(messages[1:])):
        if m['role'] == prev_m['role']:
            new_messages[-1]['content'] += m['content']
        else:
            new_messages.append(m)

        prev_m = m

    return new_messages


def format_messages(model_id: str, messages: list[dict]) -> str:
    if model_id == 'echo/echo':
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, trust_remote_code=True, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    if model_id in ('cognitivecomputations/dolphin-2.6-mistral-7b', 'NousResearch/Hermes-2-Pro-Mistral-7B', 'mtgv/MobileLLaMA-1.4B-Chat'):
        tokenizer.chat_template = CHATML_CHAT_TEMPLATE
    elif model_id in ('mistralai/Mistral-7B-Instruct-v0.2', 'NousResearch/Yarn-Mistral-7b-128k'):
        messages = create_alternate_messages(model_id, messages, convert_system_to_user=True)
    elif model_id == 'mistralai/Mixtral-8x7B-Instruct-v0.1':
        messages = create_alternate_messages(model_id, messages, convert_system_to_user=True)
    elif model_id == 'amazon/MistralLite':
        messages = create_alternate_messages(model_id, messages)
        tokenizer.chat_template = MISTRALLITE_CHAT_TEMPLATE
    elif model_id in ('microsoft/Orca-2-7b',):
        messages = create_alternate_messages(model_id, messages)
        tokenizer.chat_template = CHATML_CHAT_TEMPLATE
    elif model_id == 'GeneZC/MiniChat-2-3B':
        tokenizer.chat_template = MINICHAT_CHAT_TEMPLATE
    elif model_id == 'abacaj/phi-2-super':
        messages = create_alternate_messages(model_id, messages, convert_system_to_user=True)
    elif model_id in ('google/gemma-2b', 'google/gemma-2b-it', 'google/gemma-7b', 'google/gemma-7b-it', 'google/gemma-1.1-2b-it', 'google/gemma-1.1-7b-it'):
        messages = create_alternate_messages(model_id, messages, convert_system_to_user=True)
        # tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
    elif model_id == '01-ai/Yi-9B-200K':
        tokenizer.chat_template = CHATML_CHAT_TEMPLATE
    elif model_id == '01-ai/Yi-6B-200K':
        tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    else:
        if not tokenizer.chat_template:
            tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text


def _format_all_models_messages():
    messages = [
        {'role': 'system', 'content': 'You are a friendly chatbot.'},
        {'role': 'user', 'content': 'Hello, how are you?'},
        {'role': 'assistant', 'content': 'I\'m doing great.'},
        {'role': 'assistant', 'content': 'How can I help you today?'},
        {'role': 'user', 'content': 'I\'d like to show off how chat templating works!'},
    ]


    models = [
        # 'Qwen/Qwen1.5-14B-Chat',
        # 'Qwen/Qwen1.5-7B-Chat',
        # 'Qwen/Qwen1.5-4B-Chat',
        # 'Qwen/Qwen1.5-1.8B-Chat',
        # 'Qwen/Qwen1.5-0.5B-Chat',
        # 'cognitivecomputations/dolphin-2.7-mixtral-8x7b',
        # 'cognitivecomputations/dolphin-2.6-mistral-7b',
        'mistralai/Mistral-7B-Instruct-v0.2', # no sys, no alt
        # 'HuggingFaceH4/zephyr-7b-beta',
        # 'openchat/openchat-3.5-0106',
        # 'stabilityai/stablelm-2-zephyr-1_6b',
        # 'stabilityai/stablelm-zephyr-3b',
        # 'tiiuae/falcon-40b-instruct',
        # 'tiiuae/falcon-7b-instruct',
        # 'microsoft/Orca-2-7b', # ChatML
        # 'GeneZC/MiniChat-2-3B', # https://github.com/GeneZC/MiniMA/blob/main/conversation.py#L192
        # 'mtgv/MobileLLaMA-1.4B-Chat', # ChatML
        # 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        # 'cognitivecomputations/dolphin-2_6-phi-2',
        # 'microsoft/phi-1_5',
        # 'microsoft/phi-2',
        # 'amazon/MistralLite', # MistralLite, no alt
        # 'google/gemma-2b-it', # gemma, no alt
        # 'google/gemma-7b-it', # gemma, no alt
        'NousResearch/Hermes-2-Pro-Mistral-7B',
        'NousResearch/Yarn-Mistral-7b-128k',
    ]

    for model_id in models:
        print(f'{model_id = }')
        text: str = format_messages(model_id, messages)
        print('-' * 10)
        print(text) 
        print('-' * 10)


if __name__ == '__main__':
    _format_all_models_messages()
