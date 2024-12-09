from llama import Model


#
# first define and load/init model
#
model = Model( # 1.87B
    creator_hf_repo='vikhyatk/moondream2',
    hf_repo='vikhyatk/moondream2',
    hf_file='moondream2-text-model-f16.gguf',
    mmproj_hf_file='moondream2-mmproj-f16.gguf',
)

model.init(ctx_size=8192, predict=1024, gpu_layers=99)

#
# prompt
#
for chunk in model.completions(prompt='Describe this image.', image='examples/llama-1.png'):
    print(chunk, flush=True, end='')
else:
    print()
