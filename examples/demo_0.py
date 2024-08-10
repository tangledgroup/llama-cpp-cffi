from llama import llama_generate, get_config, Model, Options


def demo_0():
    model = Model(
        creator_hf_repo='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        hf_repo='second-state/TinyLlama-1.1B-Chat-v1.0-GGUF',
        hf_file='TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'You will write Python code in ```python...``` block.'},
        {'role': 'assistant', 'content': 'What is the task that you want to solve?'},
        {'role': 'user', 'content': 'Write Python function to evaluate expression a + b arguments, and return result.'},
    ]

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()


def demo_1():
    model = Model(
        creator_hf_repo='TinyLlama/TinyLlama_v1.1',
        hf_repo='QuantFactory/TinyLlama_v1.1-GGUF',
        hf_file='TinyLlama_v1.1.Q4_K_M.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Evaluate: 1 + 1 = ?'},
        {'role': 'assistant', 'content': '2'},
        {'role': 'user', 'content': 'Evaluate: 1 + 2 = ?'},
    ]

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()


def demo_2():
    model = Model(
        creator_hf_repo='squeeze-ai-lab/TinyAgent-1.1B',
        hf_repo='squeeze-ai-lab/TinyAgent-1.1B-GGUF',
        hf_file='TinyAgent-1.1B-Q4_K_M.gguf',
    )

    print(model)
    config = get_config(model.creator_hf_repo)

    # https://github.com/SqueezeAILab/TinyAgent/blob/0074615dd05ae632cc2321f63b4290682897334d/src/llm_compiler/planner.py#L44
    tools = []

    prefix = (
        "Given a user query, create a plan to solve it with the utmost parallelizability. "
        f"Each plan should comprise an action from the following {len(tools) + 1} types:\n"
    )

    # Tools
    for i, tool in enumerate(tools):
        prefix += f"{i+1}. {tool.description}\n"

    # Join operation
    JOIN_DESCRIPTION = (
        "join():\n"
        " - Collects and combines results from prior actions.\n"
        " - A LLM agent is called upon invoking join to either finalize the user query or wait until the plans are executed.\n"
        " - join should always be the last action in the plan, and will be called in two scenarios:\n"
        "   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.\n"
        "   (b) if the answer cannot be determined in the planning phase before you execute the plans. "
    )
    
    END_OF_PLAN = "<END_OF_PLAN>"
    JOINNER_FINISH = "Finish"
    JOINNER_REPLAN = "Replan"
    SUMMARY_RESULT = "Summary"

    prefix += f"{i+2}. {JOIN_DESCRIPTION}\n\n"

    # Guidelines
    prefix += (
        "Guidelines:\n"
        " - Each action described above contains input/output types and description.\n"
        "    - You must strictly adhere to the input and output types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        " - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.\n"
        " - Each action MUST have a unique ID, which is strictly increasing.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. "
        "In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.\n"
        f" - Always call join as the last action in the plan. Say '{END_OF_PLAN}' after you call join\n"
        " - Ensure the plan maximizes parallelizability.\n"
        " - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.\n"
        " - Never explain the plan with comments (e.g. #).\n"
        " - Never introduce new actions other than the ones provided.\n\n"
    )

    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': 'Evaluate 1 + 2. Then multiply it with 5. Result divide by 3.'},
    ]

    options = Options(
        ctx_size=config.max_position_embeddings,
        predict=-2,
        model=model,
        prompt=messages,
        stop=['<|system|>', '<|user|>', '<|assistant|>', '<|end|>'],
    )

    for chunk in llama_generate(options):
        print(chunk, flush=True, end='')

    # newline
    print()

if __name__ == '__main__':
    demo_0()
    demo_1()
    demo_2()
