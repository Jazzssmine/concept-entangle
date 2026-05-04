from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "/projects/bcxt/diff_unlearn/saeuron/style50",
    torch_dtype=torch.float16
).to("cuda")

image = pipe(
    "Image of a pony in a country fair",
    num_inference_steps=100,
    generator=torch.Generator("cuda").manual_seed(2)
).images[0]

image.save("../images/baseline_pony_country_fair.png")