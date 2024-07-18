__all__ = ['Options', 'convert_options_to_bytes']

import attr
import psutil


@attr.s
class Options:
    help = attr.ib(default=None, metadata={"description": "print usage and exit", "long_name": "--help", "alias": "-h"})
    version = attr.ib(default=None, metadata={"description": "show version and build info", "long_name": "--version"})
    verbose = attr.ib(default=None, metadata={"description": "print verbose information", "long_name": "--verbose", "alias": "-v"})
    verbosity = attr.ib(default=None, metadata={"description": "set specific verbosity level", "long_name": "--verbosity", "default": 0})
    verbose_prompt = attr.ib(default=None, metadata={"description": "print a verbose prompt before generation", "long_name": "--verbose-prompt", "default": False})
    no_display_prompt = attr.ib(default=True, metadata={"description": "don't print prompt at generation", "long_name": "--no-display-prompt", "default": False})
    color = attr.ib(default=None, metadata={"description": "colorise output to distinguish prompt and user input from generations", "long_name": "--color", "alias": "-co", "default": False})
    seed = attr.ib(default=None, metadata={"description": "RNG seed", "long_name": "--seed", "alias": "-s", "default": -1})
    threads = attr.ib(default=psutil.cpu_count(logical=False), metadata={"description": "number of threads to use during generation", "long_name": "--threads", "alias": "-t", "default": 16})
    threads_batch = attr.ib(default=None, metadata={"description": "number of threads to use during batch and prompt processing", "long_name": "--threads-batch", "alias": "-tb"})
    threads_draft = attr.ib(default=None, metadata={"description": "number of threads to use during generation", "long_name": "--threads-draft", "alias": "-td"})
    threads_batch_draft = attr.ib(default=None, metadata={"description": "number of threads to use during batch and prompt processing", "long_name": "--threads-batch-draft", "alias": "-tbd"})
    draft = attr.ib(default=None, metadata={"description": "number of tokens to draft for speculative decoding", "long_name": "--draft", "default": 5})
    p_split = attr.ib(default=None, metadata={"description": "speculative decoding split probability", "long_name": "--p-split", "alias": "-ps", "default": 0.1})
    lookup_cache_static = attr.ib(default=None, metadata={"description": "path to static lookup cache to use for lookup decoding", "long_name": "--lookup-cache-static", "alias": "-lcs"})
    lookup_cache_dynamic = attr.ib(default=None, metadata={"description": "path to dynamic lookup cache to use for lookup decoding", "long_name": "--lookup-cache-dynamic", "alias": "-lcd"})
    ctx_size = attr.ib(default=None, metadata={"description": "size of the prompt context", "long_name": "--ctx-size", "alias": "-c", "default": 0})
    predict = attr.ib(default=None, metadata={"description": "number of tokens to predict", "long_name": "--predict", "alias": "-n", "default": -1})
    batch_size = attr.ib(default=None, metadata={"description": "logical maximum batch size", "long_name": "--batch-size", "alias": "-b", "default": 2048})
    ubatch_size = attr.ib(default=None, metadata={"description": "physical maximum batch size", "long_name": "--ubatch-size", "alias": "-ub", "default": 512})
    keep = attr.ib(default=None, metadata={"description": "number of tokens to keep from the initial prompt", "long_name": "--keep", "default": 0})
    chunks = attr.ib(default=None, metadata={"description": "max number of chunks to process", "long_name": "--chunks", "default": -1})
    flash_attn = attr.ib(default=True, metadata={"description": "enable Flash Attention", "long_name": "--flash-attn", "alias": "-fa", "default": False})
    prompt = attr.ib(default=None, metadata={"description": "prompt to start generation with", "long_name": "--prompt", "alias": "-p", "default": ''})
    file = attr.ib(default=None, metadata={"description": "a file containing the prompt", "long_name": "--file", "alias": "-f"})
    in_file = attr.ib(default=None, metadata={"description": "an input file", "long_name": "--in-file"})
    binary_file = attr.ib(default=None, metadata={"description": "binary file containing the prompt", "long_name": "--binary-file", "alias": "-bf"})
    escape = attr.ib(default=None, metadata={"description": "process escapes sequences", "long_name": "--escape", "alias": "-e", "default": True})
    no_escape = attr.ib(default=None, metadata={"description": "do not process escape sequences", "long_name": "--no-escape"})
    print_token_count = attr.ib(default=None, metadata={"description": "print token count every N tokens", "long_name": "--print-token-count", "alias": "-ptc", "default": -1})
    prompt_cache = attr.ib(default=None, metadata={"description": "file to cache prompt state for faster startup", "long_name": "--prompt-cache"})
    prompt_cache_all = attr.ib(default=None, metadata={"description": "saves user input and generations to cache as well", "long_name": "--prompt-cache-all", "default": False})
    prompt_cache_ro = attr.ib(default=None, metadata={"description": "uses the prompt cache but does not update it", "long_name": "--prompt-cache-ro", "default": False})
    reverse_prompt = attr.ib(default=None, metadata={"description": "halt generation at PROMPT, return control in interactive mode", "long_name": "--reverse-prompt", "alias": "-r"})
    special = attr.ib(default=None, metadata={"description": "special tokens output enabled", "long_name": "--special", "alias": "-sp", "default": False})
    conversation = attr.ib(default=None, metadata={"description": "run in conversation mode", "long_name": "--conversation", "alias": "-cnv", "default": False})
    interactive = attr.ib(default=None, metadata={"description": "run in interactive mode", "long_name": "--interactive", "alias": "-i", "default": False})
    interactive_first = attr.ib(default=None, metadata={"description": "run in interactive mode and wait for input right away", "long_name": "--interactive-first", "alias": "-if", "default": False})
    multiline_input = attr.ib(default=None, metadata={"description": "allows you to write or paste multiple lines without ending each in '\\'", "long_name": "--multiline-input"})
    in_prefix_bos = attr.ib(default=None, metadata={"description": "prefix BOS to user inputs", "long_name": "--in-prefix-bos"})
    in_prefix = attr.ib(default=None, metadata={"description": "string to prefix user inputs with", "long_name": "--in-prefix", "default": ''})
    in_suffix = attr.ib(default=None, metadata={"description": "string to suffix after user inputs with", "long_name": "--in-suffix", "default": ''})
    spm_infill = attr.ib(default=None, metadata={"description": "use Suffix/Prefix/Middle pattern for infill", "long_name": "--spm-infill", "default": False})

    samplers = attr.ib(default=None, metadata={"description": "samplers that will be used for generation in the order", "long_name": "--samplers"})
    sampling_seq = attr.ib(default=None, metadata={"description": "simplified sequence for samplers that will be used", "long_name": "--sampling-seq", "default": "kfypmt"})
    ignore_eos = attr.ib(default=None, metadata={"description": "ignore end of stream token and continue generating", "long_name": "--ignore-eos"})
    penalize_nl = attr.ib(default=None, metadata={"description": "penalize newline tokens", "long_name": "--penalize-nl", "default": False})
    temp = attr.ib(default=None, metadata={"description": "temperature", "long_name": "--temp", "default": 0.8})
    top_k = attr.ib(default=None, metadata={"description": "top-k sampling", "long_name": "--top-k", "default": 40})
    top_p = attr.ib(default=None, metadata={"description": "top-p sampling", "long_name": "--top-p", "default": 0.9})
    min_p = attr.ib(default=None, metadata={"description": "min-p sampling", "long_name": "--min-p", "default": 0.1})
    tfs = attr.ib(default=None, metadata={"description": "tail free sampling, parameter z", "long_name": "--tfs", "default": 1.0})
    typical = attr.ib(default=None, metadata={"description": "locally typical sampling, parameter p", "long_name": "--typical", "default": 1.0})
    repeat_last_n = attr.ib(default=None, metadata={"description": "last n tokens to consider for penalize", "long_name": "--repeat-last-n", "default": 64})
    repeat_penalty = attr.ib(default=None, metadata={"description": "penalize repeat sequence of tokens", "long_name": "--repeat-penalty", "default": 1.0})
    presence_penalty = attr.ib(default=None, metadata={"description": "repeat alpha presence penalty", "long_name": "--presence-penalty", "default": 0.0})
    frequency_penalty = attr.ib(default=None, metadata={"description": "repeat alpha frequency penalty", "long_name": "--frequency-penalty", "default": 0.0})
    dynatemp_range = attr.ib(default=None, metadata={"description": "dynamic temperature range", "long_name": "--dynatemp-range", "default": 0.0})
    dynatemp_exp = attr.ib(default=None, metadata={"description": "dynamic temperature exponent", "long_name": "--dynatemp-exp", "default": 1.0})
    mirostat = attr.ib(default=None, metadata={"description": "use Mirostat sampling", "long_name": "--mirostat", "default": 0})
    mirostat_lr = attr.ib(default=None, metadata={"description": "Mirostat learning rate, parameter eta", "long_name": "--mirostat-lr", "default": 0.1})
    mirostat_ent = attr.ib(default=None, metadata={"description": "Mirostat target entropy, parameter tau", "long_name": "--mirostat-ent", "default": 5.0})
    logit_bias = attr.ib(default=None, metadata={"description": "modifies the likelihood of token appearing in the completion", "long_name": "--logit-bias", "alias": "-l"})
    cfg_negative_prompt = attr.ib(default=None, metadata={"description": "negative prompt to use for guidance", "long_name": "--cfg-negative-prompt", "default": ''})
    cfg_negative_prompt_file = attr.ib(default=None, metadata={"description": "negative prompt file to use for guidance", "long_name": "--cfg-negative-prompt-file"})
    cfg_scale = attr.ib(default=None, metadata={"description": "strength of guidance", "long_name": "--cfg-scale", "default": 1.0})
    chat_template = attr.ib(default=None, metadata={"description": "set custom jinja chat template", "long_name": "--chat-template"})

    grammar = attr.ib(default=None, metadata={"description": "BNF-like grammar to constrain generations", "long_name": "--grammar", "default": ''})
    grammar_file = attr.ib(default=None, metadata={"description": "file to read grammar from", "long_name": "--grammar-file"})
    json_schema = attr.ib(default=None, metadata={"description": "JSON schema to constrain generations", "long_name": "--json-schema", "alias": "-j"})

    pooling = attr.ib(default=None, metadata={"description": "pooling type for embeddings", "long_name": "--pooling"})
    attention = attr.ib(default=None, metadata={"description": "attention type for embeddings", "long_name": "--attention"})

    rope_scaling = attr.ib(default=None, metadata={"description": "RoPE frequency scaling method", "long_name": "--rope-scaling"})
    rope_scale = attr.ib(default=None, metadata={"description": "RoPE context scaling factor", "long_name": "--rope-scale"})
    rope_freq_base = attr.ib(default=None, metadata={"description": "RoPE base frequency", "long_name": "--rope-freq-base"})
    rope_freq_scale = attr.ib(default=None, metadata={"description": "RoPE frequency scaling factor", "long_name": "--rope-freq-scale"})
    yarn_orig_ctx = attr.ib(default=None, metadata={"description": "YaRN: original context size of model", "long_name": "--yarn-orig-ctx"})
    yarn_ext_factor = attr.ib(default=None, metadata={"description": "YaRN: extrapolation mix factor", "long_name": "--yarn-ext-factor"})
    yarn_attn_factor = attr.ib(default=None, metadata={"description": "YaRN: scale sqrt(t) or attention magnitude", "long_name": "--yarn-attn-factor", "default": 1.0})
    yarn_beta_slow = attr.ib(default=None, metadata={"description": "YaRN: high correction dim or alpha", "long_name": "--yarn-beta-slow", "default": 1.0})
    yarn_beta_fast = attr.ib(default=None, metadata={"description": "YaRN: low correction dim or beta", "long_name": "--yarn-beta-fast", "default": 32.0})
    grp_attn_n = attr.ib(default=None, metadata={"description": "group-attention factor", "long_name": "--grp-attn-n", "alias": "-gan", "default": 1})
    grp_attn_w = attr.ib(default=None, metadata={"description": "group-attention width", "long_name": "--grp-attn-w", "alias": "-gaw", "default": 512.0})
    dump_kv_cache = attr.ib(default=None, metadata={"description": "verbose print of the KV cache", "long_name": "--dump-kv-cache", "alias": "-dkvc"})
    no_kv_offload = attr.ib(default=None, metadata={"description": "disable KV offload", "long_name": "--no-kv-offload", "alias": "-nkvo"})
    cache_type_k = attr.ib(default=None, metadata={"description": "KV cache data type for K", "long_name": "--cache-type-k", "alias": "-ctk", "default": 'f16'})
    cache_type_v = attr.ib(default=None, metadata={"description": "KV cache data type for V", "long_name": "--cache-type-v", "alias": "-ctv", "default": 'f16'})

    all_logits = attr.ib(default=None, metadata={"description": "return logits for all tokens in the batch", "long_name": "--all-logits", "default": False})
    hellaswag = attr.ib(default=None, metadata={"description": "compute HellaSwag score over random tasks from datafile supplied with -f", "long_name": "--hellaswag"})
    hellaswag_tasks = attr.ib(default=None, metadata={"description": "number of tasks to use when computing the HellaSwag score", "long_name": "--hellaswag-tasks", "default": 400})
    winogrande = attr.ib(default=None, metadata={"description": "compute Winogrande score over random tasks from datafile supplied with -f", "long_name": "--winogrande"})
    winogrande_tasks = attr.ib(default=None, metadata={"description": "number of tasks to use when computing the Winogrande score", "long_name": "--winogrande-tasks", "default": 0})
    multiple_choice = attr.ib(default=None, metadata={"description": "compute multiple choice score over random tasks from datafile supplied with -f", "long_name": "--multiple-choice"})
    multiple_choice_tasks = attr.ib(default=None, metadata={"description": "number of tasks to use when computing the multiple choice score", "long_name": "--multiple-choice-tasks", "default": 0})
    kl_divergence = attr.ib(default=None, metadata={"description": "computes KL-divergence to logits provided via --kl-divergence-base", "long_name": "--kl-divergence"})
    ppl_stride = attr.ib(default=None, metadata={"description": "stride for perplexity calculation", "long_name": "--ppl-stride", "default": 0})
    ppl_output_type = attr.ib(default=None, metadata={"description": "output type for perplexity calculation", "long_name": "--ppl-output-type", "default": 0})

    defrag_thold = attr.ib(default=None, metadata={"description": "KV cache defragmentation threshold", "long_name": "--defrag-thold", "alias": "-dt", "default": -1.0})
    parallel = attr.ib(default=None, metadata={"description": "number of parallel sequences to decode", "long_name": "--parallel", "alias": "-np", "default": 1})
    sequences = attr.ib(default=None, metadata={"description": "number of sequences to decode", "long_name": "--sequences", "alias": "-ns", "default": 1})
    cont_batching = attr.ib(default=True, metadata={"description": "enable continuous batching", "long_name": "--cont-batching", "alias": "-cb", "default": True})

    mmproj = attr.ib(default=None, metadata={"description": "path to a multimodal projector file for LLaVA", "long_name": "--mmproj"})
    image = attr.ib(default=None, metadata={"description": "path to an image file", "long_name": "--image"})

    rpc = attr.ib(default=None, metadata={"description": "comma separated list of RPC servers", "long_name": "--rpc"})
    mlock = attr.ib(default=None, metadata={"description": "force system to keep model in RAM rather than swapping or compressing", "long_name": "--mlock"})
    no_mmap = attr.ib(default=None, metadata={"description": "do not memory-map model", "long_name": "--no-mmap"})
    numa = attr.ib(default=None, metadata={"description": "attempt optimizations that help on some NUMA systems", "long_name": "--numa"})
    
    gpu_layers = attr.ib(default=None, metadata={"description": "number of layers to store in VRAM", "long_name": "--gpu-layers"})
    gpu_layers_draft = attr.ib(default=None, metadata={"description": "number of layers to store in VRAM for the draft model", "long_name": "--gpu-layers-draft"})
    split_mode = attr.ib(default=None, metadata={"description": """
        how to split the model across multiple GPUs, one of:
          - none: use one GPU only
          - layer (default): split layers and KV across GPUs
          - row: split rows across GPUs
    """, "long_name": "--split-mode"})
    tensor_split = attr.ib(default=None, metadata={"description": "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1", "long_name": "--tensor-split"})
    main_gpu = attr.ib(default=None, metadata={"description": """
        the GPU to use for the model (with split-mode = none),
        or for intermediate results and KV (with split-mode = row) (default: 0)
    """, "long_name": "--main-gpu"})

    check_tensors = attr.ib(default=None, metadata={"description": "check model tensor data for invalid values", "long_name": "--check-tensors", "default": False})
    override_kv = attr.ib(default=None, metadata={"description": "advanced option to override model metadata by key", "long_name": "--override-kv"})
    lora = attr.ib(default=None, metadata={"description": "apply LoRA adapter", "long_name": "--lora"})
    lora_scaled = attr.ib(default=None, metadata={"description": "apply LoRA adapter with user defined scaling", "long_name": "--lora-scaled"})
    lora_base = attr.ib(default=None, metadata={"description": "optional model to use as a base for the layers modified by the LoRA adapter", "long_name": "--lora-base"})
    control_vector = attr.ib(default=None, metadata={"description": "add a control vector", "long_name": "--control-vector"})
    control_vector_scaled = attr.ib(default=None, metadata={"description": "add a control vector with user defined scaling", "long_name": "--control-vector-scaled"})
    control_vector_layer_range = attr.ib(default=None, metadata={"description": "layer range to apply the control vector(s) to", "long_name": "--control-vector-layer-range"})
    model = attr.ib(default=None, metadata={"description": "model path", "long_name": "--model", "alias": "-m"})
    model_draft = attr.ib(default=None, metadata={"description": "draft model for speculative decoding", "long_name": "--model-draft", "alias": "-md"})
    model_url = attr.ib(default=None, metadata={"description": "model download url", "long_name": "--model-url", "alias": "-mu"})
    # hf_repo = attr.ib(default=None, metadata={"description": "Hugging Face model repository", "long_name": "--hf-repo", "alias": "-hfr"})
    # hf_file = attr.ib(default=None, metadata={"description": "Hugging Face model file", "long_name": "--hf-file", "alias": "-hff"})

    context_file = attr.ib(default=None, metadata={"description": "file to load context from", "long_name": "--context-file"})
    chunk_size = attr.ib(default=None, metadata={"description": "minimum length of embedded text chunks", "long_name": "--chunk-size", "default": 64})
    chunk_separator = attr.ib(default=None, metadata={"description": "separator between chunks", "long_name": "--chunk-separator", "default": '\n'})

    junk = attr.ib(default=None, metadata={"description": "number of times to repeat the junk text", "long_name": "--junk", "default": 250})
    pos = attr.ib(default=None, metadata={"description": "position of the passkey in the junk text", "long_name": "--pos", "default": -1})

    output = attr.ib(default=None, metadata={"description": "output file", "long_name": "--output", "alias": "-o", "default": 'imatrix.dat'})
    output_frequency = attr.ib(default=None, metadata={"description": "output the imatrix every N iterations", "long_name": "--output-frequency", "default": 10})
    save_frequency = attr.ib(default=None, metadata={"description": "save an imatrix copy every N iterations", "long_name": "--save-frequency", "default": 0})
    process_output = attr.ib(default=None, metadata={"description": "collect data for the output tensor", "long_name": "--process-output", "default": False})
    no_ppl = attr.ib(default=None, metadata={"description": "do not compute perplexity", "long_name": "--no-ppl", "default": True})
    chunk = attr.ib(default=None, metadata={"description": "start processing the input from chunk N", "long_name": "--chunk", "default": 0})

    pps = attr.ib(default=None, metadata={"description": "is the prompt shared across parallel sequences", "long_name": "-pps"})
    npp = attr.ib(default=None, metadata={"description": "number of prompt tokens", "long_name": "-npp"})
    ntg = attr.ib(default=None, metadata={"description": "number of text generation tokens", "long_name": "-ntg"})
    npl = attr.ib(default=None, metadata={"description": "number of parallel prompts", "long_name": "-npl"})

    embd_normalize = attr.ib(default=None, metadata={"description": "normalisation for embeddings", "long_name": "--embd-normalize", "default": 2})
    embd_output_format = attr.ib(default=None, metadata={"description": "output format for embeddings", "long_name": "--embd-output-format"})
    embd_separator = attr.ib(default=None, metadata={"description": "separator of embeddings", "long_name": "--embd-separator", "default": '\n'})

    host = attr.ib(default=None, metadata={"description": "ip address to listen", "long_name": "--host", "default": '127.0.0.1'})
    port = attr.ib(default=None, metadata={"description": "port to listen", "long_name": "--port", "default": 8080})
    path = attr.ib(default=None, metadata={"description": "path to serve static files from", "long_name": "--path", "default": ''})
    embedding = attr.ib(default=None, metadata={"description": "enable embedding endpoint", "long_name": "--embedding(s)", "default": False})
    api_key = attr.ib(default=None, metadata={"description": "API key to use for authentication", "long_name": "--api-key"})
    api_key_file = attr.ib(default=None, metadata={"description": "path to file containing API keys", "long_name": "--api-key-file"})
    ssl_key_file = attr.ib(default=None, metadata={"description": "path to file a PEM-encoded SSL private key", "long_name": "--ssl-key-file"})
    ssl_cert_file = attr.ib(default=None, metadata={"description": "path to file a PEM-encoded SSL certificate", "long_name": "--ssl-cert-file"})
    timeout = attr.ib(default=None, metadata={"description": "server read/write timeout in seconds", "long_name": "--timeout", "default": 600})
    threads_http = attr.ib(default=None, metadata={"description": "number of threads used to process HTTP requests", "long_name": "--threads-http", "default": -1})
    system_prompt_file = attr.ib(default=None, metadata={"description": "set a file to load a system prompt", "long_name": "--system-prompt-file"})
    log_format = attr.ib(default=None, metadata={"description": "log output format", "long_name": "--log-format", "default": 'json'})
    metrics = attr.ib(default=None, metadata={"description": "enable prometheus compatible metrics endpoint", "long_name": "--metrics", "default": False})
    no_slots = attr.ib(default=None, metadata={"description": "disables slots monitoring endpoint", "long_name": "--no-slots", "default": True})
    slot_save_path = attr.ib(default=None, metadata={"description": "path to save slot kv cache", "long_name": "--slot-save-path"})
    slot_prompt_similarity = attr.ib(default=None, metadata={"description": "how much the prompt of a request must match the prompt of a slot in order to use that slot", "long_name": "--slot-prompt-similarity", "alias": "-sps", "default": 0.50})

    simple_io = attr.ib(default=True, metadata={"description": "use basic IO for better compatibility in subprocesses and limited consoles", "long_name": "--simple-io"})
    logdir = attr.ib(default=None, metadata={"description": "path under which to save YAML logs", "long_name": "--logdir", "alias": "-ld"})
    log_test = attr.ib(default=None, metadata={"description": "Run simple logging test", "long_name": "--log-test"})
    log_disable = attr.ib(default=True, metadata={"description": "Disable trace logs", "long_name": "--log-disable"})
    log_enable = attr.ib(default=None, metadata={"description": "Enable trace logs", "long_name": "--log-enable"})
    log_file = attr.ib(default=None, metadata={"description": "Specify a log filename", "long_name": "--log-file"})
    log_new = attr.ib(default=None, metadata={"description": "Create a separate new log file on start", "long_name": "--log-new"})
    log_append = attr.ib(default=None, metadata={"description": "Don't truncate the old log file", "long_name": "--log-append"})

    output = attr.ib(default=None, metadata={"description": "output file", "long_name": "--output", "alias": "-o", "default": 'control_vector.gguf'})
    positive_file = attr.ib(default=None, metadata={"description": "positive prompts file", "long_name": "--positive-file", "default": 'examples/cvector-generator/positive.txt'})
    negative_file = attr.ib(default=None, metadata={"description": "negative prompts file", "long_name": "--negative-file", "default": 'examples/cvector-generator/negative.txt'})
    pca_batch = attr.ib(default=None, metadata={"description": "batch size used for PCA", "long_name": "--pca-batch", "default": 100})
    pca_iter = attr.ib(default=None, metadata={"description": "number of iterations used for PCA", "long_name": "--pca-iter", "default": 1000})
    method = attr.ib(default=None, metadata={"description": "dimensionality reduction method to be used", "long_name": "--method", "default": 'pca'})


def convert_options_to_bytes(options: Options) -> list[bytes]:
    result = []

    # Iterate over all attributes of the options class
    for field in attr.fields(Options):
        value = getattr(options, field.name)
        if value is not None:
            long_name = field.metadata["long_name"]
            alias = field.metadata.get("alias")

            # Handle boolean options
            if isinstance(value, bool):
                if value:
                    result.append(long_name.encode())
            # Handle other options
            else:
                result.append(long_name.encode())
                result.append(str(value).encode())

    return result
