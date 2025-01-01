from llama import Model

#
# first define and load/init model
#
# https://huggingface.co/vikhyatk/moondream2
# Apache license 2.0
model = Model( # 1.87B
    creator_hf_repo='vikhyatk/moondream2',
    hf_repo='vikhyatk/moondream2',
    hf_file='moondream2-text-model-f16.gguf',
    mmproj_hf_file='moondream2-mmproj-f16.gguf',
)

model.init(ctx_size=8 * 1024, gpu_layers=99)

#
# prompt
#
prompt = 'Describe this image.'
image = 'examples/llama-1.png'

completions = model.completions(
    prompt=prompt,
    image=image,
    predict=1 * 1024,
)

for chunk in completions:
    print(chunk, flush=True, end='')

print()
