from diffusers import StableDiffusionPipeline
import torch
import os
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from I2P dataset")

    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_checkpoint", type=str, help="Path to erased unet checkpoint", default="")
    parser.add_argument("--prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--prompt_file", type=str, help="Optional path to a file containing prompts (one per line or bucket|prompt format)", default="")
    parser.add_argument("--num_images", type=int, help="Number of images to generate for testing", default=10)
    parser.add_argument('--num_inference_steps', help='num_inference_steps', type=int, required=False, default=50)
    parser.add_argument('--guidance_scale', help='guidance_scale', type=float, required=False, default=7.5)
    args = parser.parse_args()
    return args


def sanitize_filename(text):
    text = "_".join(text.split())
    text = re.sub(r"[^A-Za-z0-9_]", "", text)
    return text[:200] if text else "prompt"


def load_prompts(prompt_file):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Supports "bucket|prompt" while preserving plain one-line prompt files.
            if "|" in line:
                bucket, prompt = line.split("|", 1)
                bucket = bucket.strip()
                prompt = prompt.strip()
                if prompt:
                    prompts.append((prompt, bucket if bucket else None))
            else:
                prompts.append((line, None))
    return prompts


if __name__ == "__main__":
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = torch.Generator(device)

    if not args.prompt and not args.prompt_file:
        raise ValueError("Provide either --prompt or --prompt_file.")
    if args.prompt and args.prompt_file:
        raise ValueError("Use only one of --prompt or --prompt_file.")

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, safety_checker=None, torch_dtype=torch.float16).to(device)

    if(args.unet_checkpoint != ""):
        print("Loading erased unet checkpoint from ", args.unet_checkpoint)
        pipe.unet.load_state_dict(torch.load(args.unet_checkpoint))

    if args.prompt_file:
        prompts_to_generate = load_prompts(args.prompt_file)
        if len(prompts_to_generate) == 0:
            raise ValueError(f"No prompts found in file: {args.prompt_file}")
    else:
        prompts_to_generate = [(args.prompt, None)]

    with torch.no_grad():
        for prompt, bucket in prompts_to_generate:
            save_dir = args.output_dir
            if bucket:
                save_dir = os.path.join(args.output_dir, sanitize_filename(bucket))
                os.makedirs(save_dir, exist_ok=True)

            filename = sanitize_filename(prompt)
            for i in range(args.num_images):
                gen.manual_seed(i)
                torch.manual_seed(i)
                out = pipe(prompt=[prompt], generator=gen, num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
                image = out.images[0]
                image.save(os.path.join(save_dir, f"{filename}_{i}.png"))
 