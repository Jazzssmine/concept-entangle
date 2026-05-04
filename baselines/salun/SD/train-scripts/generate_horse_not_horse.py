import argparse
import csv
import sys
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


HORSE_PROMPT_TEMPLATES = [
    "a photo of a horse",
    "a realistic horse standing in a field",
    "a close-up portrait of a horse",
    "a horse running on grass",
    "a horse in natural daylight",
    "a horse near a wooden fence",
    "a horse on a farm",
    "a horse by the countryside road",
    "a horse in front of mountains",
    "a horse in a meadow",
]

NOT_HORSE_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a rabbit",
    "a photo of a cow",
    "a photo of a sheep",
    "a photo of a deer",
    "a photo of a fish underwater",
    "a photo of a butterfly on a flower",
    "a photo of a city street",
]

CASTLE_PROMPT_TEMPLATES = [
    "a photo of a castle",
    "a medieval castle on a hill",
    "a stone castle surrounded by fog",
    "a grand castle reflected in a lake",
    "a castle gate with old stone walls",
    "a castle with towers and battlements",
    "a castle above a river valley",
    "a castle courtyard in daylight",
    "a castle near a forest at sunset",
    "a realistic castle in a mountain landscape",
]

NOT_CASTLE_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a car",
    "a photo of a truck",
    "a photo of a bridge",
    "a photo of a city street",
    "a photo of a forest path",
    "a photo of a mountain lake",
    "a photo of a beach at sunset",
]

TOWER_PROMPT_TEMPLATES = [
    "a photo of a tower",
    "a stone tower on a hill",
    "a tall watchtower against the sky",
    "a historic tower reflected in a river",
    "a tower with old brick walls",
    "a tower rising above city rooftops",
    "a tower in a mountain valley",
    "a tower plaza in daylight",
    "a tower near a forest at sunset",
    "a realistic tower in a dramatic landscape",
]

NOT_TOWER_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a car",
    "a photo of a truck",
    "a photo of a bridge",
    "a photo of a city street",
    "a photo of a forest path",
    "a photo of a mountain lake",
    "a photo of a beach at sunset",
]

BEAR_PROMPT_TEMPLATES = [
    "a photo of a bear",
    "a realistic bear standing in a forest",
    "a close-up portrait of a bear",
    "a bear walking through a meadow",
    "a bear near a river in daylight",
    "a brown bear in a mountain landscape",
    "a bear in the wild",
    "a bear near pine trees",
    "a bear on a rocky trail",
    "a bear in natural habitat",
]

NOT_BEAR_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a car",
    "a photo of a truck",
    "a photo of a bridge",
    "a photo of a city street",
    "a photo of a forest path",
    "a photo of a mountain lake",
    "a photo of a beach at sunset",
]

DOG_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a realistic dog sitting on grass",
    "a close-up portrait of a dog",
    "a dog running in a park",
    "a dog in natural daylight",
    "a dog near a wooden fence",
    "a dog on a trail",
    "a dog by a lake",
    "a dog looking at the camera",
    "a dog in a backyard",
]

NOT_DOG_PROMPT_TEMPLATES = [
    "a photo of a cat",
    "a photo of a horse",
    "a photo of a bear",
    "a photo of a car",
    "a photo of a truck",
    "a photo of a bridge",
    "a photo of a city street",
    "a photo of a forest path",
    "a photo of a mountain lake",
    "a photo of a beach at sunset",
]

CAT_PROMPT_TEMPLATES = [
    "a photo of a cat",
    "a realistic cat sitting on a windowsill",
    "a close-up portrait of a cat",
    "a cat lying on a sofa",
    "a cat in natural daylight",
    "a cat walking in a garden",
    "a cat indoors on a wooden floor",
    "a cat looking at the camera",
    "a cat near a houseplant",
    "a cat resting on a blanket",
]

NOT_CAT_PROMPT_TEMPLATES = [
    "a photo of a dog",
    "a photo of a horse",
    "a photo of a bird",
    "a photo of a rabbit",
    "a photo of a cow",
    "a photo of a sheep",
    "a photo of a deer",
    "a photo of a fish underwater",
    "a photo of a butterfly on a flower",
    "a photo of a city street",
]

PROMPT_SETS = {
    "horse": (HORSE_PROMPT_TEMPLATES, NOT_HORSE_PROMPT_TEMPLATES),
    "castle": (CASTLE_PROMPT_TEMPLATES, NOT_CASTLE_PROMPT_TEMPLATES),
    "tower": (TOWER_PROMPT_TEMPLATES, NOT_TOWER_PROMPT_TEMPLATES),
    "bear": (BEAR_PROMPT_TEMPLATES, NOT_BEAR_PROMPT_TEMPLATES),
    "dog": (DOG_PROMPT_TEMPLATES, NOT_DOG_PROMPT_TEMPLATES),
    "cat": (CAT_PROMPT_TEMPLATES, NOT_CAT_PROMPT_TEMPLATES),
}


def load_model_from_config(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_one_image(model, sampler, prompt, device, image_size, ddim_steps, guidance_scale, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    uc = model.get_learned_conditioning([""])
    cond = model.get_learned_conditioning([prompt])
    latent_shape = [4, image_size // 8, image_size // 8]
    start_code = torch.randn((1, *latent_shape), generator=generator, device=device)

    latents, _ = sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        batch_size=1,
        shape=latent_shape,
        unconditional_guidance_scale=guidance_scale,
        unconditional_conditioning=uc,
        eta=0.0,
        x_T=start_code,
        verbose=False,
    )
    decoded = model.decode_first_stage(latents)
    decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    image = rearrange(decoded[0].cpu().numpy(), "c h w -> h w c")
    image = (image * 255.0).round().astype("uint8")
    return Image.fromarray(image)


def generate_dataset(
    model,
    sampler,
    prompts,
    output_dir,
    count,
    seed_start,
    device,
    image_size,
    ddim_steps,
    guidance_scale,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "seed", "prompt"])

        for idx in tqdm(range(count), desc=f"Generating {output_dir.name}"):
            prompt = prompts[idx % len(prompts)]
            seed = seed_start + idx
            image = sample_one_image(
                model=model,
                sampler=sampler,
                prompt=prompt,
                device=device,
                image_size=image_size,
                ddim_steps=ddim_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            filename = f"{idx:04d}.png"
            image.save(output_dir / filename)
            writer.writerow([idx, filename, seed, prompt])


def generate_dataset_diffusers(
    pipe,
    prompts,
    output_dir,
    count,
    seed_start,
    image_size,
    ddim_steps,
    guidance_scale,
    generator_device,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"

    with metadata_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename", "seed", "prompt"])

        for idx in tqdm(range(count), desc=f"Generating {output_dir.name}"):
            prompt = prompts[idx % len(prompts)]
            seed = seed_start + idx
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                height=image_size,
                width=image_size,
                num_inference_steps=ddim_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            filename = f"{idx:04d}.png"
            image.save(output_dir / filename)
            writer.writerow([idx, filename, seed, prompt])


def main():
    parser = argparse.ArgumentParser(
        description="Generate concept and non-concept image sets for SalUn training."
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="horse",
        choices=["horse", "castle", "tower", "bear", "dog", "cat"],
        help="Target concept to generate.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional HF diffusers model id/path (e.g. CompVis/stable-diffusion-v1-4).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/work/hdd/bcxt/anon3/unlearn_diff/stable-diffusion/sd-v1-4-full-ema.ckpt",
        help="Path to SD1.4 compvis checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="Path to SD config yaml.",
    )
    parser.add_argument(
        "--horse_dir",
        type=str,
        default="/work/hdd/bcxt/anon3/unlearn_diff/salun/SD/data/horse",
        help="Output directory for horse images.",
    )
    parser.add_argument(
        "--not_horse_dir",
        type=str,
        default="/work/hdd/bcxt/anon3/unlearn_diff/salun/SD/data/not-horse",
        help="Output directory for non-horse images.",
    )
    parser.add_argument("--count", type=int, default=800, help="Images per folder.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--image_size", type=int, default=512, help="Image resolution.")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    args = parser.parse_args()

    if args.model_path:
        try:
            from diffusers import DDIMScheduler, StableDiffusionPipeline
        except ImportError as e:
            raise ImportError(
                "diffusers is required for --model_path. Install with: pip install diffusers"
            ) from e

        dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(args.device)
        pipe.set_progress_bar_config(disable=True)

        generator_device = "cuda" if args.device.startswith("cuda") else "cpu"
        target_prompts, non_target_prompts = PROMPT_SETS[args.concept]

        generate_dataset_diffusers(
            pipe=pipe,
            prompts=target_prompts,
            output_dir=Path(args.horse_dir),
            count=args.count,
            seed_start=args.seed,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            generator_device=generator_device,
        )
        generate_dataset_diffusers(
            pipe=pipe,
            prompts=non_target_prompts,
            output_dir=Path(args.not_horse_dir),
            count=args.count,
            seed_start=args.seed + 100000,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
            generator_device=generator_device,
        )
    else:
        target_prompts, non_target_prompts = PROMPT_SETS[args.concept]

        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

        model = load_model_from_config(
            config_path=str(config_path),
            ckpt_path=args.ckpt_path,
            device=args.device,
        )
        sampler = DDIMSampler(model)

        generate_dataset(
            model=model,
            sampler=sampler,
            prompts=target_prompts,
            output_dir=Path(args.horse_dir),
            count=args.count,
            seed_start=args.seed,
            device=args.device,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )
        generate_dataset(
            model=model,
            sampler=sampler,
            prompts=non_target_prompts,
            output_dir=Path(args.not_horse_dir),
            count=args.count,
            seed_start=args.seed + 100000,
            device=args.device,
            image_size=args.image_size,
            ddim_steps=args.ddim_steps,
            guidance_scale=args.guidance_scale,
        )

    print("Done.")
    print(f"{args.concept.capitalize()} images: {args.horse_dir}")
    print(f"Not-{args.concept} images: {args.not_horse_dir}")


if __name__ == "__main__":
    main()
