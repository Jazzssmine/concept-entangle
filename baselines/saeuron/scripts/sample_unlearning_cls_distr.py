import csv
import os
import pickle
import sys
from pathlib import Path

# Add parent directory to path to allow imports of utils and SAE
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from packaging import version
from tqdm import tqdm

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

import fire

from UnlearnCanvas_resources.const import (
    class_available,
    theme_available,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

from diffusers.utils.import_utils import is_xformers_available


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_sae(sae_checkpoint, hookpoint, device):
    sae = Sae.load_from_disk(
        os.path.join(sae_checkpoint, hookpoint), device=device
    ).eval()
    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False
    return sae


def main(
    pipe_checkpoint,
    hookpoint,
    class_latents_path,
    sae_checkpoint,
    # class_params_path,
    target_class=None,
    prompt=None,
    prompts_file=None,
    percentile=None,
    multiplier=None,
    seed=188,
    steps=100,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/class20/",
):
    # Configure Accelerator to avoid multi-node issues if MASTER_ADDR is not set
    # When accelerate launch is used, it may detect distributed mode but MASTER_ADDR might not be set
    # Set it to localhost for single-node multi-GPU setups
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    # Only set MASTER_PORT if not already set
    # Use a port based on a hash of the output_dir to ensure consistency across processes
    # but avoid conflicts between different runs
    if "MASTER_PORT" not in os.environ:
        import hashlib
        port_base = 29500
        # Generate a deterministic port based on output_dir to avoid conflicts
        port_hash = int(hashlib.md5(output_dir.encode()).hexdigest()[:4], 16) % 1000
        os.environ["MASTER_PORT"] = str(port_base + port_hash)
    
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)

    if is_xformers_available():
        import xformers

        if accelerator.is_main_process:
            print("Enabling xFormers memory efficient attention")
        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            if accelerator.is_main_process:
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
        model.enable_xformers_memory_efficient_attention()

    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    with open(
        class_latents_path,
        "rb",
    ) as f:
        class_latents_dict = pickle.load(f)

    # class_params = torch.load(class_params_path)
    if prompt is not None and prompts_file is not None:
        raise ValueError("Use either 'prompt' or 'prompts_file', not both.")

    custom_prompts = []
    if prompts_file is not None:
        ppath = Path(prompts_file)
        if ppath.suffix.lower() == ".csv":
            with open(prompts_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or ()
                if "prompt" not in fields:
                    raise ValueError(
                        "CSV prompts_file must include a 'prompt' column; "
                        f"found columns: {list(fields)}"
                    )
                for row in reader:
                    text = (row.get("prompt") or "").strip()
                    if text:
                        custom_prompts.append(text)
        else:
            with open(prompts_file, "r", encoding="utf-8") as f:
                custom_prompts = [line.strip() for line in f if line.strip()]
    elif prompt is not None:
        custom_prompts = [prompt]

    run_custom_prompts = len(custom_prompts) > 0
    custom_percentile = 99.995 if percentile is None else float(percentile)
    custom_multiplier = -25.0 if multiplier is None else float(multiplier)
    default_percentile = 99.99 if percentile is None else float(percentile)
    default_multiplier = -1.0 if multiplier is None else float(multiplier)

    theme_avail = [t for t in theme_available if t != "Seed_Images"]
    if target_class is not None:
        if target_class not in class_available:
            raise ValueError(
                f"Invalid target_class '{target_class}'. Must be one of: {class_available}"
            )
        classes_to_unlearn = [target_class]
    else:
        classes_to_unlearn = class_available

    progress_bar = tqdm(
        classes_to_unlearn,
        total=len(classes_to_unlearn),
        disable=not accelerator.is_main_process,
    )
    for class_to_unlearn in progress_bar:
        if accelerator.is_main_process:
            progress_bar.set_description(f"Unlearning {class_to_unlearn}")
        output_path = os.path.join(
            output_dir,
            f"{class_to_unlearn}",
        )
        os.makedirs(output_path, exist_ok=True)
        if run_custom_prompts:
            prompt_items = list(enumerate(custom_prompts))
            with accelerator.split_between_processes(prompt_items) as local_prompt_items:
                local_prompt_indices = [idx for idx, _ in local_prompt_items]
                local_prompts = [txt for _, txt in local_prompt_items]
                steering_hooks = {}
                steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                    concept_to_unlearn=[class_to_unlearn],
                    percentile=custom_percentile,
                    multiplier=custom_multiplier,
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=class_latents_dict,
                    sae=sae,
                    steps=steps,
                    preserve_error=True,
                )
                if len(local_prompts) > 0:
                    with torch.no_grad():
                        images = model.run_with_hooks(
                            prompt=local_prompts,
                            generator=generator,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            position_hook_dict=steering_hooks,
                        )
                else:
                    images = []
            accelerator.wait_for_everyone()
            images = gather_object(images)
            prompt_indices = gather_object(local_prompt_indices)
            if accelerator.is_main_process:
                for img, idx in zip(images, prompt_indices):
                    img.save(
                        os.path.join(
                            output_path,
                            f"custom_{idx:03d}_seed{seed}.jpg",
                        )
                    )
            accelerator.wait_for_everyone()
            continue

        for test_theme in theme_avail:
            input_classes = []
            input_themes = []
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            with accelerator.split_between_processes(
                class_theme_pairs
            ) as local_classes_themes:
                local_prompts = []
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                steering_hooks = {}
                steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                    concept_to_unlearn=[class_to_unlearn],
                    percentile=default_percentile,
                    multiplier=default_multiplier,
                    feature_importance_fn=compute_feature_importance,
                    concept_latents_dict=class_latents_dict,
                    sae=sae,
                    steps=steps,
                    preserve_error=True,
                )
                with torch.no_grad():
                    images = model.run_with_hooks(
                        prompt=local_prompts,
                        generator=generator,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        position_hook_dict=steering_hooks,
                    )
                for object_class, theme in local_classes_themes:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            if accelerator.is_main_process:
                for img, object_class, theme in zip(
                    images, input_classes, input_themes
                ):
                    if theme == "":
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}.jpg",
                            )
                        )
                    else:
                        img.save(
                            os.path.join(
                                output_path,
                                f"{theme}_{object_class}_seed{seed}.jpg",
                            )
                        )
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    fire.Fire(main)
