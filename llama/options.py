__all__ = ['Options', 'convert_options_to_bytes']


from typing import Optional

import psutil
from attrs import define, field, fields


@define
class Options:
    # General options
    help: Optional[bool] = field(default=None, metadata={"long_name": "--help", "description": "Print usage and exit"})
    version: Optional[bool] = field(default=None, metadata={"long_name": "--version", "description": "Show version and build info"})
    verbose: Optional[bool] = field(default=None, metadata={"long_name": "--verbose", "description": "Print verbose information"})
    verbosity: Optional[int] = field(default=None, metadata={"long_name": "--verbosity", "description": "Set specific verbosity level (default: 0)"})
    verbose_prompt: Optional[bool] = field(default=None, metadata={"long_name": "--verbose-prompt", "description": "Print a verbose prompt before generation (default: false)"})
    no_display_prompt: Optional[bool] = field(default=True, metadata={"long_name": "--no-display-prompt", "description": "Don't print prompt at generation (default: false)"})
    color: Optional[bool] = field(default=None, metadata={"long_name": "--color", "description": "Colorize output to distinguish prompt and user input from generations (default: false)"})
    seed: Optional[int] = field(default=None, metadata={"long_name": "--seed", "description": "RNG seed (default: -1, use random seed for < 0)"})
    threads: Optional[int] = field(default=psutil.cpu_count(logical=False), metadata={"long_name": "--threads", "description": "Number of threads to use during generation (default: 16)"})
    threads_batch: Optional[int] = field(default=None, metadata={"long_name": "--threads-batch", "description": "Number of threads to use during batch and prompt processing (default: same as --threads)"})
    threads_draft: Optional[int] = field(default=None, metadata={"long_name": "--threads-draft", "description": "Number of threads to use during generation (default: same as --threads)"})
    threads_batch_draft: Optional[int] = field(default=None, metadata={"long_name": "--threads-batch-draft", "description": "Number of threads to use during batch and prompt processing (default: same as --threads-draft)"})
    draft: Optional[int] = field(default=None, metadata={"long_name": "--draft", "description": "Number of tokens to draft for speculative decoding (default: 5)"})
    p_split: Optional[float] = field(default=None, metadata={"long_name": "--p-split", "description": "Speculative decoding split probability (default: 0.1)"})
    lookup_cache_static: Optional[str] = field(default=None, metadata={"long_name": "--lookup-cache-static", "description": "Path to static lookup cache to use for lookup decoding (not updated by generation)"})
    lookup_cache_dynamic: Optional[str] = field(default=None, metadata={"long_name": "--lookup-cache-dynamic", "description": "Path to dynamic lookup cache to use for lookup decoding (updated by generation)"})
    ctx_size: Optional[int] = field(default=None, metadata={"long_name": "--ctx-size", "description": "Size of the prompt context (default: 0, 0 = loaded from model)"})
    predict: Optional[int] = field(default=None, metadata={"long_name": "--predict", "description": "Number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)"})
    batch_size: Optional[int] = field(default=None, metadata={"long_name": "--batch-size", "description": "Logical maximum batch size (default: 2048)"})
    ubatch_size: Optional[int] = field(default=None, metadata={"long_name": "--ubatch-size", "description": "Physical maximum batch size (default: 512)"})
    keep: Optional[int] = field(default=None, metadata={"long_name": "--keep", "description": "Number of tokens to keep from the initial prompt (default: 0, -1 = all)"})
    chunks: Optional[int] = field(default=None, metadata={"long_name": "--chunks", "description": "Max number of chunks to process (default: -1, -1 = all)"})
    flash_attn: Optional[bool] = field(default=None, metadata={"long_name": "--flash-attn", "description": "Enable Flash Attention (default: disabled)"})
    prompt: Optional[str] = field(default=None, metadata={"long_name": "--prompt", "description": "Prompt to start generation with"})
    file: Optional[str] = field(default=None, metadata={"long_name": "--file", "description": "A file containing the prompt (default: none)"})
    in_file: Optional[str] = field(default=None, metadata={"long_name": "--in-file", "description": "An input file (repeat to specify multiple files)"})
    binary_file: Optional[str] = field(default=None, metadata={"long_name": "--binary-file", "description": "Binary file containing the prompt (default: none)"})
    escape: Optional[bool] = field(default=None, metadata={"long_name": "--escape", "description": "Process escape sequences (default: true)"})
    no_escape: Optional[bool] = field(default=None, metadata={"long_name": "--no-escape", "description": "Do not process escape sequences"})
    print_token_count: Optional[int] = field(default=None, metadata={"long_name": "--print-token-count", "description": "Print token count every N tokens (default: -1)"})
    prompt_cache: Optional[str] = field(default=None, metadata={"long_name": "--prompt-cache", "description": "File to cache prompt state for faster startup (default: none)"})
    prompt_cache_all: Optional[bool] = field(default=None, metadata={"long_name": "--prompt-cache-all", "description": "Saves user input and generations to cache as well"})
    prompt_cache_ro: Optional[bool] = field(default=None, metadata={"long_name": "--prompt-cache-ro", "description": "Uses the prompt cache but does not update it"})
    reverse_prompt: Optional[str] = field(default=None, metadata={"long_name": "--reverse-prompt", "description": "Halt generation at PROMPT, return control in interactive mode"})
    special: Optional[bool] = field(default=None, metadata={"long_name": "--special", "description": "Special tokens output enabled (default: false)"})
    conversation: Optional[bool] = field(default=None, metadata={"long_name": "--conversation", "description": "Run in conversation mode (default: false)"})
    interactive: Optional[bool] = field(default=None, metadata={"long_name": "--interactive", "description": "Run in interactive mode (default: false)"})
    interactive_first: Optional[bool] = field(default=None, metadata={"long_name": "--interactive-first", "description": "Run in interactive mode and wait for input right away (default: false)"})
    multiline_input: Optional[bool] = field(default=None, metadata={"long_name": "--multiline-input", "description": "Allows you to write or paste multiple lines without ending each in '\\'"})
    in_prefix_bos: Optional[bool] = field(default=None, metadata={"long_name": "--in-prefix-bos", "description": "Prefix BOS to user inputs, preceding the `--in-prefix` string"})
    in_prefix: Optional[str] = field(default=None, metadata={"long_name": "--in-prefix", "description": "String to prefix user inputs with (default: empty)"})
    in_suffix: Optional[str] = field(default=None, metadata={"long_name": "--in-suffix", "description": "String to suffix after user inputs with (default: empty)"})
    no_warmup: Optional[bool] = field(default=None, metadata={"long_name": "--no-warmup", "description": "Skip warming up the model with an empty run"})
    spm_infill: Optional[bool] = field(default=None, metadata={"long_name": "--spm-infill", "description": "Use Suffix/Prefix/Middle pattern for infill (default: disabled)"})

    # Sampling options
    samplers: Optional[str] = field(default=None, metadata={"long_name": "--samplers", "description": "Samplers that will be used for generation in the order, separated by ';'"})
    sampling_seq: Optional[str] = field(default=None, metadata={"long_name": "--sampling-seq", "description": "Simplified sequence for samplers that will be used (default: kfypmt)"})
    ignore_eos: Optional[bool] = field(default=None, metadata={"long_name": "--ignore-eos", "description": "Ignore end of stream token and continue generating"})
    penalize_nl: Optional[bool] = field(default=None, metadata={"long_name": "--penalize-nl", "description": "Penalize newline tokens (default: false)"})
    temp: Optional[float] = field(default=None, metadata={"long_name": "--temp", "description": "Temperature (default: 0.8)"})
    top_k: Optional[int] = field(default=None, metadata={"long_name": "--top-k", "description": "Top-k sampling (default: 40, 0 = disabled)"})
    top_p: Optional[float] = field(default=None, metadata={"long_name": "--top-p", "description": "Top-p sampling (default: 0.9, 1.0 = disabled)"})
    min_p: Optional[float] = field(default=None, metadata={"long_name": "--min-p", "description": "Min-p sampling (default: 0.1, 0.0 = disabled)"})
    tfs: Optional[float] = field(default=None, metadata={"long_name": "--tfs", "description": "Tail free sampling, parameter z (default: 1.0, 1.0 = disabled)"})
    typical: Optional[float] = field(default=None, metadata={"long_name": "--typical", "description": "Locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)"})
    repeat_last_n: Optional[int] = field(default=None, metadata={"long_name": "--repeat-last-n", "description": "Last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)"})
    repeat_penalty: Optional[float] = field(default=None, metadata={"long_name": "--repeat-penalty", "description": "Penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)"})
    presence_penalty: Optional[float] = field(default=None, metadata={"long_name": "--presence-penalty", "description": "Repeat alpha presence penalty (default: 0.0, 0.0 = disabled)"})
    frequency_penalty: Optional[float] = field(default=None, metadata={"long_name": "--frequency-penalty", "description": "Repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)"})
    dynatemp_range: Optional[float] = field(default=None, metadata={"long_name": "--dynatemp-range", "description": "Dynamic temperature range (default: 0.0, 0.0 = disabled)"})
    dynatemp_exp: Optional[float] = field(default=None, metadata={"long_name": "--dynatemp-exp", "description": "Dynamic temperature exponent (default: 1.0)"})
    mirostat: Optional[int] = field(default=None, metadata={"long_name": "--mirostat", "description": "Use Mirostat sampling (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"})
    mirostat_lr: Optional[float] = field(default=None, metadata={"long_name": "--mirostat-lr", "description": "Mirostat learning rate, parameter eta (default: 0.1)"})
    mirostat_ent: Optional[float] = field(default=None, metadata={"long_name": "--mirostat-ent", "description": "Mirostat target entropy, parameter tau (default: 5.0)"})
    logit_bias: Optional[str] = field(default=None, metadata={"long_name": "--logit-bias", "description": "Modifies the likelihood of token appearing in the completion"})
    cfg_negative_prompt: Optional[str] = field(default=None, metadata={"long_name": "--cfg-negative-prompt", "description": "Negative prompt to use for guidance (default: '')"})
    cfg_negative_prompt_file: Optional[str] = field(default=None, metadata={"long_name": "--cfg-negative-prompt-file", "description": "Negative prompt file to use for guidance"})
    cfg_scale: Optional[float] = field(default=None, metadata={"long_name": "--cfg-scale", "description": "Strength of guidance (default: 1.0, 1.0 = disable)"})
    chat_template: Optional[str] = field(default=None, metadata={"long_name": "--chat-template", "description": "Set custom jinja chat template"})

    # Grammar options
    grammar: Optional[str] = field(default=None, metadata={"long_name": "--grammar", "description": "BNF-like grammar to constrain generations (default: '')"})
    grammar_file: Optional[str] = field(default=None, metadata={"long_name": "--grammar-file", "description": "File to read grammar from"})
    json_schema: Optional[str] = field(default=None, metadata={"long_name": "--json-schema", "description": "JSON schema to constrain generations"})

    # Embedding options
    pooling: Optional[str] = field(default=None, metadata={"long_name": "--pooling", "description": "Pooling type for embeddings"})
    attention: Optional[str] = field(default=None, metadata={"long_name": "--attention", "description": "Attention type for embeddings"})

    # Context hacking options
    rope_scaling: Optional[str] = field(default=None, metadata={"long_name": "--rope-scaling", "description": "RoPE frequency scaling method"})
    rope_scale: Optional[float] = field(default=None, metadata={"long_name": "--rope-scale", "description": "RoPE context scaling factor"})
    rope_freq_base: Optional[float] = field(default=None, metadata={"long_name": "--rope-freq-base", "description": "RoPE base frequency"})
    rope_freq_scale: Optional[float] = field(default=None, metadata={"long_name": "--rope-freq-scale", "description": "RoPE frequency scaling factor"})
    yarn_orig_ctx: Optional[int] = field(default=None, metadata={"long_name": "--yarn-orig-ctx", "description": "YaRN: original context size of model"})
    yarn_ext_factor: Optional[float] = field(default=None, metadata={"long_name": "--yarn-ext-factor", "description": "YaRN: extrapolation mix factor"})
    yarn_attn_factor: Optional[float] = field(default=None, metadata={"long_name": "--yarn-attn-factor", "description": "YaRN: scale sqrt(t) or attention magnitude"})
    yarn_beta_slow: Optional[float] = field(default=None, metadata={"long_name": "--yarn-beta-slow", "description": "YaRN: high correction dim or alpha"})
    yarn_beta_fast: Optional[float] = field(default=None, metadata={"long_name": "--yarn-beta-fast", "description": "YaRN: low correction dim or beta"})
    grp_attn_n: Optional[int] = field(default=None, metadata={"long_name": "--grp-attn-n", "description": "Group-attention factor"})
    grp_attn_w: Optional[float] = field(default=None, metadata={"long_name": "--grp-attn-w", "description": "Group-attention width"})
    dump_kv_cache: Optional[bool] = field(default=None, metadata={"long_name": "--dump-kv-cache", "description": "Verbose print of the KV cache"})
    no_kv_offload: Optional[bool] = field(default=None, metadata={"long_name": "--no-kv-offload", "description": "Disable KV offload"})
    cache_type_k: Optional[str] = field(default=None, metadata={"long_name": "--cache-type-k", "description": "KV cache data type for K (default: f16)"})
    cache_type_v: Optional[str] = field(default=None, metadata={"long_name": "--cache-type-v", "description": "KV cache data type for V (default: f16)"})

    # Perplexity options
    all_logits: Optional[bool] = field(default=None, metadata={"long_name": "--all-logits", "description": "Return logits for all tokens in the batch (default: false)"})
    hellaswag: Optional[bool] = field(default=None, metadata={"long_name": "--hellaswag", "description": "Compute HellaSwag score over random tasks"})
    hellaswag_tasks: Optional[int] = field(default=None, metadata={"long_name": "--hellaswag-tasks", "description": "Number of tasks to use when computing the HellaSwag score (default: 400)"})
    winogrande: Optional[bool] = field(default=None, metadata={"long_name": "--winogrande", "description": "Compute Winogrande score over random tasks"})
    winogrande_tasks: Optional[int] = field(default=None, metadata={"long_name": "--winogrande-tasks", "description": "Number of tasks to use when computing the Winogrande score (default: 0)"})
    multiple_choice: Optional[bool] = field(default=None, metadata={"long_name": "--multiple-choice", "description": "Compute multiple choice score over random tasks"})
    multiple_choice_tasks: Optional[int] = field(default=None, metadata={"long_name": "--multiple-choice-tasks", "description": "Number of tasks to use when computing the multiple choice score (default: 0)"})
    kl_divergence: Optional[bool] = field(default=None, metadata={"long_name": "--kl-divergence", "description": "Computes KL-divergence to logits provided via --kl-divergence-base"})
    ppl_stride: Optional[int] = field(default=None, metadata={"long_name": "--ppl-stride", "description": "Stride for perplexity calculation (default: 0)"})
    ppl_output_type: Optional[int] = field(default=None, metadata={"long_name": "--ppl-output-type", "description": "Output type for perplexity calculation (default: 0)"})

    # Parallel options
    defrag_thold: Optional[float] = field(default=None, metadata={"long_name": "--defrag-thold", "description": "KV cache defragmentation threshold (default: -1.0, < 0 - disabled)"})
    parallel: Optional[int] = field(default=None, metadata={"long_name": "--parallel", "description": "Number of parallel sequences to decode (default: 1)"})
    sequences: Optional[int] = field(default=None, metadata={"long_name": "--sequences", "description": "Number of sequences to decode (default: 1)"})
    cont_batching: Optional[bool] = field(default=None, metadata={"long_name": "--cont-batching", "description": "Enable continuous batching (default: enabled)"})
    no_cont_batching: Optional[bool] = field(default=None, metadata={"long_name": "--no-cont-batching", "description": "Disable continuous batching"})

    # Multi-modality options
    mmproj: Optional[str] = field(default=None, metadata={"long_name": "--mmproj", "description": "Path to a multimodal projector file for LLaVA"})
    image: Optional[str] = field(default=None, metadata={"long_name": "--image", "description": "Path to an image file"})

    # Backend options
    rpc: Optional[str] = field(default=None, metadata={"long_name": "--rpc", "description": "Comma-separated list of RPC servers"})
    mlock: Optional[bool] = field(default=None, metadata={"long_name": "--mlock", "description": "Force system to keep model in RAM"})
    no_mmap: Optional[bool] = field(default=None, metadata={"long_name": "--no-mmap", "description": "Do not memory-map model"})
    numa: Optional[str] = field(default=None, metadata={"long_name": "--numa", "description": "Attempt optimizations that help on some NUMA systems"})
    gpu_layers: Optional[int] = field(default=None, metadata={"long_name": "--gpu-layers", "description": "Number of layers to store in VRAM"})
    gpu_layers_draft: Optional[int] = field(default=None, metadata={"long_name": "--gpu-layers-draft", "description": "Number of layers to store in VRAM for the draft model"})
    split_mode: Optional[str] = field(default=None, metadata={"long_name": "--split-mode", "description": """
        how to split the model across multiple GPUs, one of:
          - none: use one GPU only
          - layer (default): split layers and KV across GPUs
          - row: split rows across GPUs
    """})
    tensor_split: Optional[str] = field(default=None, metadata={"long_name": "--tensor-split", "description": "Fraction of the model to offload to each GPU"})
    main_gpu: Optional[int] = field(default=None, metadata={"long_name": "--main-gpu", "description": """
        the GPU to use for the model (with split-mode = none),
        or for intermediate results and KV (with split-mode = row) (default: 0)
    """})

    # Model options
    check_tensors: Optional[bool] = field(default=None, metadata={"long_name": "--check-tensors", "description": "Check model tensor data for invalid values (default: false)"})
    override_kv: Optional[str] = field(default=None, metadata={"long_name": "--override-kv", "description": "Advanced option to override model metadata by key"})
    lora: Optional[str] = field(default=None, metadata={"long_name": "--lora", "description": "Apply LoRA adapter (can be repeated to use multiple adapters)"})
    lora_scaled: Optional[str] = field(default=None, metadata={"long_name": "--lora-scaled", "description": "Apply LoRA adapter with user-defined scaling"})
    control_vector: Optional[str] = field(default=None, metadata={"long_name": "--control-vector", "description": "Add a control vector"})
    control_vector_scaled: Optional[str] = field(default=None, metadata={"long_name": "--control-vector-scaled", "description": "Add a control vector with user-defined scaling"})
    control_vector_layer_range: Optional[str] = field(default=None, metadata={"long_name": "--control-vector-layer-range", "description": "Layer range to apply the control vector(s) to"})
    model: Optional[str] = field(default=None, metadata={"long_name": "--model", "description": "Model path (default: models/$filename with filename from --hf-file or --model-url)"})
    model_draft: Optional[str] = field(default=None, metadata={"long_name": "--model-draft", "description": "Draft model for speculative decoding (default: unused)"})

    # Logging options
    simple_io: Optional[bool] = field(default=True, metadata={"long_name": "--simple-io", "description": "Use basic IO for better compatibility in subprocesses"})
    logdir: Optional[str] = field(default=None, metadata={"long_name": "--logdir", "description": "Path under which to save YAML logs (no logging if unset)"})
    log_test: Optional[bool] = field(default=None, metadata={"long_name": "--log-test", "description": "Run simple logging test"})
    log_disable: Optional[bool] = field(default=True, metadata={"long_name": "--log-disable", "description": "Disable trace logs"})
    log_enable: Optional[bool] = field(default=None, metadata={"long_name": "--log-enable", "description": "Enable trace logs"})
    log_file: Optional[str] = field(default=None, metadata={"long_name": "--log-file", "description": "Specify a log filename"})
    log_new: Optional[bool] = field(default=None, metadata={"long_name": "--log-new", "description": "Create a separate new log file on start"})
    log_append: Optional[bool] = field(default=None, metadata={"long_name": "--log-append", "description": "Don't truncate the old log file"})

    # Non-llama-cli fields
    stop: Optional[list[str]] = field(default=None, metadata={"long_name": "--stop", "description": "Stop"})


def convert_options_to_bytes(options: Options) -> list[bytes]:
    result = []

    for field in fields(Options):
        # skip non-llama-cli fields
        if field.name in ('stop',):
            continue

        value = getattr(options, field.name)
        
        if value is not None:
            long_name = field.metadata['long_name']
            alias = field.metadata.get('alias')

            if isinstance(value, bool):
                # handle boolean options
                if value:
                    result.append(long_name.encode())
            else:
                # handle other options
                result.append(long_name.encode())
                result.append(str(value).encode())

    return result
