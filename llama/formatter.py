__all__ = [
    'CHATML_CHAT_TEMPLATE',
    'ZEPHYR_CHAT_TEMPLATE',
    'SYSTEM_USER_ASSISTANT_TEMPLATE',
    'VLM_TEMPLATE',
    'create_alternate_messages',
    'get_fallback_chat_template',
    'get_config',
    'get_tokenizer',
    'get_special_tokens',
    'format_messages',
    'AutoConfig',
]

import json
from copy import deepcopy
from typing import Optional
from collections import namedtuple

# import jinja2
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer

from .options import CompletionsOptions


FALLBACK_MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'


CHATML_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

ZEPHYR_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|' + message['role'] + '|>\n' + message['content'] + '<|end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
    "{% endif %}"
)

SYSTEM_USER_ASSISTANT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ 'System: ' + message['content'] }}\n\n"
        "{% elif message['role'] == 'user' %}"
            "{{ 'User: ' + message['content'] }}\n\n"
        "{% elif message['role'] == 'assistant' %}"
            "{{ 'Assistant: '  + message['content'] }}\n\n"
        "{% endif %}"

        "{% if loop.last and add_generation_prompt %}"
            "{{ 'Assistant:' }}"
        "{% endif %}"
    "{% endfor %}"
)

VLM_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n"
        "{% elif message['role'] == 'user' %}"
            "{% for item in message['content'] %}"
                "{% if item['type'] == 'image_url' %}"
                    "{{ '<image>\n' }}"
                "{% elif item['type'] == 'text' %}"
                    "{{ item['text'] }}\n"
                "{% endif %}"
            "{% endfor %}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}\n"
        "{% endif %}"

        "{% if loop.last and add_generation_prompt %}"
            "{{ '' }}"
        "{% endif %}"
    "{% endfor %}"
)


def create_alternate_messages(messages: list[dict], convert_system_to_user: bool=False) -> list[dict]:
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
            new_messages[-1]['content'] += ' ' + m['content']
        else:
            new_messages.append(m)

        prev_m = m

    return new_messages


def get_fallback_chat_template(tokenizer: AutoTokenizer) -> str:
    special_tokens: list[str] = get_special_tokens(tokenizer)

    if '<|im_end|>' in special_tokens:
        return CHATML_CHAT_TEMPLATE
    else:
        return ZEPHYR_CHAT_TEMPLATE


def get_config(model_id: str) -> AutoConfig:
    try:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        config_path = hf_hub_download(repo_id=model_id, filename='config.json')

        with open(config_path) as f:
            config_data: dict = json.load(f)

        _ModelConfig = namedtuple('_ModelConfig', [k for k in config_data.keys() if not k.startswith('_')])
        config = _ModelConfig(**{k: v for k, v in config_data.items() if not k.startswith('_')})

    return config


def get_special_tokens(tokenizer: AutoTokenizer, force_standard_special_tokens: bool=False) -> list[str]:
    special_tokens: set[str] | list[str] = []

    if force_standard_special_tokens:
        special_tokens += [
            # mixed
            '<s>',
            '</s>',
            '[INST]',
            '[/INST]',
            '[SYS]',
            '[/SYS]',
            '[AVAILABLE_TOOLS]',
            '[/AVAILABLE_TOOLS]',
            '<<INST>>',
            '<</INST>>',
            '<<SYS>>',
            '<</SYS>>',
            '[UNUSED_TOKEN_145]',
            '<|startoftext|>',
            '<|endoftext|>',
            '<|system|>',
            '<|user|>',
            '<|assistant|>',
            '<|tool|>',
            '<|end|>',

            # chatml
            '<|im_start|>',
            '<|im_end|>',

            # llama 3
            '<|begin_of_text|>',
            '<|end_of_text|>',
            '<|start_header_id|>',
            '<|end_header_id|>',
            '<|eot_id|>',
        ]
    else:
        special_tokens += tokenizer.all_special_tokens + tokenizer.additional_special_tokens # type: ignore
        assert isinstance(special_tokens, list)

        if '<s>' in special_tokens and '</s>' not in special_tokens:
            special_tokens.append('</s>')

        if '<s>' in special_tokens and '[AVAILABLE_TOOLS]' not in special_tokens:
            special_tokens.append('[AVAILABLE_TOOLS]')

        if '<s>' in special_tokens and '[/AVAILABLE_TOOLS]' not in special_tokens:
            special_tokens.append('[/AVAILABLE_TOOLS]')

        if '<s>' in special_tokens and '[INST]' not in special_tokens:
            special_tokens.append('[INST]')

        if '<s>' in special_tokens and '[/INST]' not in special_tokens:
            special_tokens.append('[/INST]')

        if '<s>' in special_tokens and '<<INST>>' not in special_tokens:
            special_tokens.append('<<INST>>')

        if '<s>' in special_tokens and '<</INST>>' not in special_tokens:
            special_tokens.append('<</INST>>')

        if '<s>' in special_tokens and '<<SYS>>' not in special_tokens:
            special_tokens.append('<<SYS>>')

        if '<s>' in special_tokens and '<</SYS>>' not in special_tokens:
            special_tokens.append('<</SYS>>')

        if '<s>' in special_tokens and '[SYS]' not in special_tokens:
            special_tokens.append('[SYS]')

        if '<s>' in special_tokens and '[/SYS]' not in special_tokens:
            special_tokens.append('[/SYS]')

        if '<s>' in special_tokens and '[UNUSED_TOKEN_145]' not in special_tokens:
            special_tokens.append('[UNUSED_TOKEN_145]')

        if '<|startoftext|>' in special_tokens and '<|endoftext|>' not in special_tokens:
            special_tokens.append('<|endoftext|>')

        if '<|endoftext|>' in special_tokens and '<|end|>' not in special_tokens:
            special_tokens.append('<|end|>')

        if '<|endoftext|>' in special_tokens and '<|system|>' not in special_tokens:
            special_tokens.append('<|system|>')

        if '<|endoftext|>' in special_tokens and '<|user|>' not in special_tokens:
            special_tokens.append('<|user|>')

        if '<|endoftext|>' in special_tokens and '<|assistant|>' not in special_tokens:
            special_tokens.append('<|assistant|>')

        if '<|im_end|>' in special_tokens and '<|im_start|>' not in special_tokens:
            special_tokens.append('<|im_start|>')

        special_tokens = set(special_tokens)
        special_tokens = list(special_tokens)
        special_tokens.sort()

    # print(f'{special_tokens = }')
    return special_tokens


def get_tokenizer(model_id: str) -> AutoTokenizer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_ID, trust_remote_code=True, use_fast=True)
        tokenizer.chat_template = get_fallback_chat_template(tokenizer)

    if not tokenizer.chat_template:
        tokenizer.chat_template = get_fallback_chat_template(tokenizer)

    return tokenizer


def format_messages(tokenizer: AutoTokenizer, messages: list[dict], completions_options: Optional[CompletionsOptions]=None) -> str:
    if completions_options and completions_options.chat_template:
        tokenizer.chat_template = completions_options.chat_template # type: ignore
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    else:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        except Exception:
            messages = create_alternate_messages(messages, convert_system_to_user=True)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore

    # print(f'{text = }')
    return text
