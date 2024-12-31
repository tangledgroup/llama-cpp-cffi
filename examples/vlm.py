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

model.init(ctx_size=8192, gpu_layers=99)

#
# prompt
#
for chunk in model.completions(prompt='Describe this image.', image='examples/llama-1.png', predict=1024):
    print(chunk, flush=True, end='')

print()
