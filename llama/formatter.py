__all__ = ['get_config', 'get_tokenizer', 'get_special_tokens', 'format_messages', 'AutoConfig']

from copy import deepcopy

import jinja2
from transformers import AutoConfig, AutoTokenizer


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
            new_messages[-1]['content'] += '\n' + m['content']
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
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return config


def get_special_tokens(tokenizer: AutoTokenizer, force_standard_special_tokens: bool=False) -> list[str]:
    special_tokens: list[str] = []
    
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
        special_tokens += tokenizer.all_special_tokens + tokenizer.additional_special_tokens

        if '<s>' in special_tokens and not '</s>' in special_tokens:
            special_tokens.append('</s>')

        if '<s>' in special_tokens and not '[AVAILABLE_TOOLS]' in special_tokens:
            special_tokens.append('[AVAILABLE_TOOLS]')

        if '<s>' in special_tokens and not '[/AVAILABLE_TOOLS]' in special_tokens:
            special_tokens.append('[/AVAILABLE_TOOLS]')

        if '<s>' in special_tokens and not '[INST]' in special_tokens:
            special_tokens.append('[INST]')

        if '<s>' in special_tokens and not '[/INST]' in special_tokens:
            special_tokens.append('[/INST]')

        if '<s>' in special_tokens and not '<<INST>>' in special_tokens:
            special_tokens.append('<<INST>>')

        if '<s>' in special_tokens and not '<</INST>>' in special_tokens:
            special_tokens.append('<</INST>>')

        if '<s>' in special_tokens and not '<<SYS>>' in special_tokens:
            special_tokens.append('<<SYS>>')

        if '<s>' in special_tokens and not '<</SYS>>' in special_tokens:
            special_tokens.append('<</SYS>>')

        if '<s>' in special_tokens and not '[SYS]' in special_tokens:
            special_tokens.append('[SYS]')

        if '<s>' in special_tokens and not '[/SYS]' in special_tokens:
            special_tokens.append('[/SYS]')

        if '<s>' in special_tokens and not '[UNUSED_TOKEN_145]' in special_tokens:
            special_tokens.append('[UNUSED_TOKEN_145]')

        if '<|startoftext|>' in special_tokens and not '<|endoftext|>' in special_tokens:
            special_tokens.append('<|endoftext|>')
        
        if '<|endoftext|>' in special_tokens and not '<|end|>' in special_tokens:
            special_tokens.append('<|end|>')

        if '<|endoftext|>' in special_tokens and not '<|system|>' in special_tokens:
            special_tokens.append('<|system|>')

        if '<|endoftext|>' in special_tokens and not '<|user|>' in special_tokens:
            special_tokens.append('<|user|>')

        if '<|endoftext|>' in special_tokens and not '<|assistant|>' in special_tokens:
            special_tokens.append('<|assistant|>')
        
        if '<|im_end|>' in special_tokens and not '<|im_start|>' in special_tokens:
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


def format_messages(tokenizer: AutoTokenizer, messages: list[dict]) -> str:
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        messages = create_alternate_messages(messages, convert_system_to_user=True)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # print(f'{text = }')
    return text
