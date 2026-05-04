# Erasing Concepts from Diffusion Models
###  [Project Website](https://erasing.baulab.info) | [Arxiv Preprint](https://arxiv.org/pdf/2303.07345.pdf) | [Fine-tuned Weights](https://erasing.baulab.info/weights/esd_models/) | [Demo](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion) <br>

### Updated code 🚀 - Now support diffusers!! (Faster and cleaner)

<div align='center'>
<img src = 'images/applications.png'>
</div>

## Code Update 🚀
We are releasing a cleaner code for ESD with diffusers support. Compared to our old-code this version uses almost half the GPU memory and is 5-8 times faster. Because of this diffusers support - we believe it allows to generalise to latest models (FLUX ESD coming soon ... ) <br>

To use the older version please visit our [older commit](https://github.com/rohitgandikota/erasing/tree/a2189e9ae677aca22a00c361bde25d3d320d8a61)

## Installation Guide
We recently updated our codebase to be much more cleaner and faster. The setup is also simple
```bash
git clone https://github.com/rohitgandikota/erasing.git
cd erasing
pip install -r requirements.txt
```

for FLUX training:

```bash
pip install --upgrade diffusers transformers torch
```

## Training Guide

### SDv1.4
After installation, follow these instructions to train a custom ESD model for Stable Diffusion V1.4. Pick from following `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'`:
```python
python esd_sd.py --erase_concept 'Van Gogh' --train_method 'esd-x'
python esd_sd.py --erase_concept 'bear' --train_method 'esd-x' --iterations 500 --negative_guidance 5 --guidance_scale 7
```

💡 New application: You can now erase an attribute from a concept!! Instead of erasing a whole concept you can just precisely remove some of its attributes. For example, you can erase hats from cowboys but keep the rest intact!
```python
python esd_sd.py --erase_concept 'cowboy hat' --erase_from 'cowboy' --train_method 'xattn'
```

### SDXL
After installation, follow these instructions to train a custom ESD model for Stable Diffusion V1.4. Pick from following `'esd-x'`, `'esd-x-strict'` (NOTE: `'esd-u'` is currently experimental and might produce unexpected artifacts):
```python
python esd_sdxl.py --erase_concept 'Van Gogh' --train_method 'esd-x-strict'
```

### FLUX
After installation (make sure: diffusers and transformers are up-to-date), follow these instructions to train a custom ESD model for FLUX.1-dev. Pick from following `'esd-x'`, `'esd-x-strict'` [ONLY WORKS ON 80GB GPUs - if you can do it in less memory, please open a PR, I would love to learn your tricks]:
```python
python esd_flux.py --erase_concept 'monster' --train_method 'esd-x' --negative_guidance 2
```

The optimization process for erasing undesired visual concepts from pre-trained diffusion model weights involves using a short text description of the concept as guidance. The ESD model is fine-tuned with the conditioned and unconditioned scores obtained from frozen SD model to guide the output away from the concept being erased. The model learns from it's own knowledge to steer the diffusion process away from the undesired concept.
<div align='center'>
<img src = 'images/ESD.png'>
</div>

## Generating Images

Generating images from custom ESD model is super easy. Please follow `notebook/esd_inference_sdxl.ipynb` notebook

For an automated script to generate a ton of images for your evaluations use our evalscripts
```python
python evalscripts/generate-images.py --base_model 'CompVis/stable-diffusion-v1-4' --esd_path 'esd-models/sd/esd-bicycle-from-bicycle-esdx.safetensors' --prompts_path './data/human_bike_prompts.csv' --num_inference_steps 20 --guidance_scale 7

python evalscripts/generate-images.py --base_model 'CompVis/stable-diffusion-v1-4' --esd_path 'esd-models/sd/esd-horses-from-horses-esdx.safetensors' --prompts_path './data/human_horse_prompts.csv' --num_inference_steps 20 --guidance_scale 7

python evalscripts/generate-images.py \
  --base_model 'CompVis/stable-diffusion-v1-4' \
  --prompts_path ./data/gibberish_prompts.csv \
  --num_inference_steps 50 \
  --guidance_scale 7 \
  --save_path ./esd-images/original_gibberish \
  --num_samples 50

python evalscripts/generate-images.py \
  --base_model 'CompVis/stable-diffusion-v1-4' \
  --prompts_path ./data/bagofwords_gibberish.txt \
  --num_inference_steps 50 \
  --guidance_scale 7 \
  --save_path ./esd-images/original_bagofwords_gibberish \
  --txt_file \
  --num_samples 5

python evalscripts/generate-images.py \
  --base_model 'CompVis/stable-diffusion-v1-4' \
  --esd_path 'esd-models/sd/esd-horses-from-horses-esdx.safetensors' \
  --prompts_path ./data/single.csv \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --save_path ./esd-images/single_esd \
  --num_samples 5
```

```
baseline
python evalscripts/generate-images.py \
  --base_model 'CompVis/stable-diffusion-v1-4' \
  --prompts_path ./data/single.csv \
  --save_path ./esd-images/single \
  --num_inference_steps 50 --guidance_scale 7.5\
  --num_samples 5

unlearned
python evalscripts/generate-images.py --base_model 'CompVis/stable-diffusion-v1-4' --esd_path 'esd-models/sd/esd-bicycle-from-bicycle-esdx.safetensors' --prompts_path './data/bicycle_only_prompts.csv' --num_inference_steps 20 --guidance_scale 7
```

### UPDATE (NudeNet)
If you want to recreate the results from our paper on NSFW task - please use this https://drive.google.com/file/d/1J_O-yZMabmS9gBA2qsFSrmFoCl7tbFgK/view?usp=sharing

* Untar this file and save it in the homedirectory '~/.NudeNet'
* This should enable the right results as we use this checkpoint for our analysis.

## Running Gradio Demo Locally

To run the gradio interactive demo locally, clone the files from [demo repository](https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/tree/main) <br>

* Create an environment using the packages included in the requirements.txt file
* Run `python app.py`
* Open the application in browser at `http://127.0.0.1:7860/`
* Train, evaluate, and save models using our method

## NOTE ON LICENSE
The code and methods behind our work have been released under MIT. However, the models that you use our methods with, might be on a different licenses. Please read the model's license (the model you are using) carefully for more details. 

## Citing our work
The preprint can be cited as follows
```
@inproceedings{gandikota2023erasing,
  title={Erasing Concepts from Diffusion Models},
  author={Rohit Gandikota and Joanna Materzy\'nska and Jaden Fiotto-Kaufman and David Bau},
  booktitle={Proceedings of the 2023 IEEE International Conference on Computer Vision},
  year={2023}
}
```
