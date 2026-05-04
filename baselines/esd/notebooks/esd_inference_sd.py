import os 
import torch
import random
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file
from copy import deepcopy
torch.set_grad_enabled(False)

torch_dtype = torch.bfloat16
device = 'cuda:0'
basemodel_id="CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=torch_dtype, use_safetensors=True, safety_checker=None).to(device)
original_weights = deepcopy(pipe.unet.state_dict())
esd_weights = load_file("../esd-models/sd/esd-Van_Gogh-from-Van_Gogh-esdx.safetensors")

num_inference_steps = 20
guidance_scale = 7.5
height=width=512

# Generate with original model
prompt = 'image of starry night in Van Gogh style'
seed = random.randint(0, 2**15)


pipe.unet.load_state_dict(original_weights, strict=False)
image = pipe(prompt, 
             num_inference_steps = num_inference_steps,
             guidance_scale= guidance_scale,
             height=height,
             width=width,
             generator=torch.Generator().manual_seed(seed)
            ).images
original_vangogh = image[0].resize((256, 256))
original_vangogh.save("original_starry_night.png")

# Generate with ESD model
pipe.unet.load_state_dict(esd_weights, strict=False)
image = pipe(prompt, 
             num_inference_steps = num_inference_steps,
             guidance_scale= guidance_scale,
             height=height,
             width=width,
             generator=torch.Generator().manual_seed(seed)
            ).images
esd_vangogh = image[0].resize((256, 256))
esd_vangogh.save("esd_starry_night.png")