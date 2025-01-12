__all__ = ['ModelOptions', 'CompletionsOptions']

from typing import Optional, ForwardRef

import psutil
from attrs import define, field

from .llama_cpp import ggml_type


Model = ForwardRef('Model')

LLAMA_DEFAULT_SEED = 0xFFFFFFFF
LLAMA_SPLIT_MODE_NONE = 0 # single GPU
LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
LLAMA_SPLIT_MODE_ROW = 2 # split layers and KV across GPUs, use tensor parallelism if supported


@define
class ModelOptions:
    creator_hf_repo: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_file: Optional[str] = None
    mmproj_hf_file: Optional[str] = None
    tokenizer_hf_repo: Optional[str] = None

    verbose: bool = False # Set verbosity level to infinity (i.e. log all messages, useful for debugging)
    n_threads: int = psutil.cpu_count(logical=False) or 1 # number of threads to use during generation (default: -1)
    n_threads_batch: Optional[int] = None # number of threads for batch and prompt processing (default: same as --threads)
    # cpu_mask: str = field(default="") # CPU affinity mask: arbitrarily long hex (default: "")
    # cpu_range: Optional[str] = field(default=None) # range of CPUs for affinity (default: "")
    # cpu_strict: int = field(default=0) # use strict CPU placement (default: 0)
    # prio: int = field(default=0) # set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: 0)
    # poll: int = field(default=50) # use polling level to wait for work (0 - no polling, default: 50)
    # cpu_mask_batch: Optional[str] = field(default=None) # CPU affinity mask for batch processing (default: same as --cpu-mask)
    # cpu_range_batch: Optional[str] = field(default=None) # ranges of CPUs for batch affinity (default: "")
    # cpu_strict_batch: Optional[int] = field(default=None) # use strict CPU placement for batch (default: same as --cpu-strict)
    # prio_batch: int = field(default=0) # set process/thread priority for batch (default: 0)
    # poll_batch: Optional[int] = field(default=None) # use polling to wait for work for batch (default: same as --poll)
    n_ctx: int = 4096 # size of the prompt context (default: 4096, 0 = loaded from model)
    n_batch: int = 2048 # logical maximum batch size (default: 2048)
    n_ubatch: int = 512 # physical maximum batch size (default: 512)
    # keep: int = field(default=0) # number of tokens to keep from initial prompt (default: 0, -1 = all)
    flash_attn: bool = False # enable Flash Attention (default: disabled)
    # no_perf: bool = field(default=False) # disable internal libllama performance timings (default: false)
    # rope_scaling: Optional[str] = field(default=None) # RoPE frequency scaling method (default: linear unless specified by model)
    # rope_scale: Optional[float] = field(default=None) # RoPE context scaling factor, expands context by factor of N
    # rope_freq_base: Optional[float] = field(default=None) # RoPE base frequency, used by NTK-aware scaling (default: loaded from model)
    # rope_freq_scale: Optional[float] = field(default=None) # RoPE frequency scaling factor, expands context by factor of 1/N
    # yarn_orig_ctx: Optional[int] = field(default=None) # YaRN: original context size of model (default: 0 = model training context size)
    # yarn_ext_factor: float = field(default=-1.0) # YaRN: extrapolation mix factor (default: -1.0, 0.0 = full interpolation)
    # yarn_attn_factor: float = field(default=1.0) # YaRN: scale sqrt(t) or attention magnitude (default: 1.0)
    # yarn_beta_slow: float = field(default=1.0) # YaRN: high correction dim or alpha (default: 1.0)
    # yarn_beta_fast: float = field(default=32.0) # YaRN: low correction dim or beta (default: 32.0)
    # dump_kv_cache: bool = field(default=False) # verbose print of the KV cache
    # no_kv_offload: bool = field(default=False) # disable KV offload
    type_k: ggml_type = ggml_type.F16 # KV cache data type for K (default: f16)
    type_v: ggml_type = ggml_type.F16 # KV cache data type for V (default: f16)
    # defrag_thold: float = field(default=0.1) # KV cache defragmentation threshold (default: 0.1, < 0 - disabled)
    # parallel: int = field(default=1) # number of parallel sequences to decode (default: 1)
    use_mlock: bool = False # force system to keep model in RAM rather than swapping or compressing
    use_mmap: bool = True # memory-map model (slower load but may reduce pageouts if not using mlock)
    # numa: Optional[str] = field(default=None) # attempt optimizations for NUMA systems
    # device: Optional[str] = field(default=None) # comma-separated list of devices to use for offloading (default: none)
    # list_devices: bool = field(default=False) # print list of available devices and exit
    gpu_layers: int = 0 # number of layers to store in VRAM
    split_mode: int = 0 # how to split the model across multiple GPUs
    # tensor_split: Optional[str] = field(default=None) # fraction of the model to offload to each GPU
    main_gpu: int = 0 # the GPU to use for the model or intermediate results and KV
    check_tensors: bool = False # check model tensor data for invalid values (default: false)
    # override_kv: List[str] = field(factory=list) # advanced option to override model metadata by key
    # lora: List[str] = field(factory=list) # path to LoRA adapter (can be repeated)
    # lora_scaled: List[Tuple[str, float]] = field(factory=list) # path to LoRA adapter with user-defined scaling (can be repeated)
    # control_vector: List[str] = field(factory=list) # add a control vector (can be repeated)
    # control_vector_scaled: List[Tuple[str, float]] = field(factory=list) # add a control vector with user-defined scaling (can be repeated)
    # control_vector_layer_range: Optional[Tuple[int, int]] = field(default=None) # layer range to apply control vector(s) to

    # Example-specific parameters
    # no_display_prompt: bool = field(default=False) # don't print prompt at generation (default: false)
    # no_context_shift: bool = field(default=False) # disables context shift on infinite text generation (default: disabled)
    # print_token_count: int = field(default=-1) # print token count every N tokens (default: -1)
    # prompt_cache: Optional[str] = field(default=None) # file to cache prompt state for faster startup (default: none)
    # prompt_cache_all: bool = field(default=False) # if specified, saves user input and generations to cache as well
    # prompt_cache_ro: bool = field(default=False) # if specified, uses the prompt cache but does not update it
    # special: bool = field(default=False) # special tokens output enabled (default: false)
    # no_warmup: bool = field(default=False) # skip warming up the model with an empty run
    # grp_attn_n: int = field(default=1) # group-attention factor (default: 1)
    # grp_attn_w: int = field(default=512) # group-attention width (default: 512)


@define
class CompletionsOptions:
    verbose: bool = False # Set verbosity level to infinity (i.e. log all messages, useful for debugging)
    prompt: Optional[str] = None # prompt to start generation with
    messages: Optional[list[dict[str, str]]] = None # prompt messages to start generation with
    image: Optional[str] = None # Path to an image file

    # TODO: predict: int = field(default=-1) # number of tokens to predict (default: -1, -1 = infinity, -2 = until context filled)
    predict: int = 512 # number of tokens to predict
    # file: Optional[str] = field(default=None)  # a file containing the prompt (default: none)

    # Sampling parameters
    # samplers: Optional[str] = field(default=None) # samplers for generation in order, separated by ';' (default: dry;top_k;typ_p;top_p;min_p;xtc;temperature)
    seed: int = LLAMA_DEFAULT_SEED # RNG seed (default: -1, use random seed for -1)
    # sampling_seq: Optional[str] = field(default=None) # simplified sequence for samplers
    temp: float = 0.8 # temperature (default: 0.8)
    top_k: int = 40 # top-k sampling (default: 40, 0 = disabled)
    top_p: float = 0.9 # top-p sampling (default: 0.9, 1.0 = disabled)
    min_p: float = 0.1 # min-p sampling (default: 0.1, 0.0 = disabled)
    # xtc_probability: float = field(default=0.0) # xtc probability (default: 0.0, 0.0 = disabled)
    # xtc_threshold: float = field(default=0.1) # xtc threshold (default: 0.1, 1.0 = disabled)
    # typical: float = field(default=1.0) # locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)
    repeat_last_n: int = 64 # last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)
    repeat_penalty: float = 1.0 # penalize repeat sequence of tokens (default: 1.0, 1.0 = disabled)
    frequency_penalty: float = 0.0 # repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)
    presence_penalty: float = 0.0 # repeat alpha presence penalty (default: 0.0, 0.0 = disabled...
    penalize_nl: bool = False # penalize newline tokens (default: false)
    ignore_eos: bool = False # ignore end of stream token and continue generating
    dry_multiplier: float = 0.0 # set DRY sampling multiplier (default: 0.0, 0.0 = disabled)
    dry_base: float = 1.75 # set DRY sampling base value (default: 1.75)
    dry_allowed_length: int = 2 # set allowed length for DRY sampling (default: 2)
    dry_penalty_last_n: int = -1 # set DRY penalty for the last n tokens (default: -1, 0 = disable, -1 = context size)
    dry_sequence_breaker: list[str] = field(factory=lambda: []) # add sequence breaker for DRY sampling, clearing out default breakers
                                                                # ('\n', ':', '"', '*') in the process; use "none" to not use any
                                                                # sequence breakers
    # dynatemp_range: float = field(default=0.0) # dynamic temperature range (default: 0.0, 0.0 = disabled)
    # dynatemp_exp: float = field(default=1.0) # dynamic temperature exponent (default: 1.0)
    # mirostat: int = field(default=0) # use Mirostat sampling (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    # mirostat_lr: float = field(default=0.1) # Mirostat learning rate, parameter eta (default: 0.1)
    # mirostat_ent: float = field(default=5.0) # Mirostat target entropy, parameter tau (default: 5.0)
    # logit_bias: List[str] = field(factory=list) # modifies likelihood of token appearing in completion
    grammar: Optional[str | bytes] = None # BNF-like grammar to constrain generations (default: '')
    # grammar_file: Optional[str] = field(default=None) # file to read grammar from
    json_schema: Optional[str | bytes | dict] = None # JSON schema to constrain generations
    chat_template: Optional[str] = None # set custom jinja chat template (default: from model's metadata)
