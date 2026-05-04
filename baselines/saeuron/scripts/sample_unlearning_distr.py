import os
import pickle
import sys
import time
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
    style_latents_path,
    sae_checkpoint,
    seed=188,
    steps=100,
    percentile=99.999,
    multiplier=-1.0,
    guidance_scale=9.0,
    output_dir="eval_results/mu_results/style50/",
    batch_size=None,  # If None, process all prompts at once. If set, process in batches of this size.
    themes_to_unlearn=None,  # If None, unlearn all themes. If provided as a list, unlearn only those specific themes.
):
    # Set up distributed training environment variables if not already set
    # This is needed when running with accelerate launch in SLURM
    if "MASTER_ADDR" not in os.environ:
        # Try to get from SLURM environment
        if "SLURM_JOB_NODELIST" in os.environ:
            # Extract first node from SLURM node list
            nodelist = os.environ["SLURM_JOB_NODELIST"]
            # Simple extraction - get first node name
            first_node = nodelist.split(",")[0].split("[")[0]
            os.environ.setdefault("MASTER_ADDR", first_node)
        else:
            os.environ.setdefault("MASTER_ADDR", "localhost")
    
    if "MASTER_PORT" not in os.environ:
        os.environ.setdefault("MASTER_PORT", "12355")
    
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
    
    if "LOCAL_RANK" not in os.environ:
        os.environ.setdefault("LOCAL_RANK", "0")
    
    if "WORLD_SIZE" not in os.environ:
        os.environ.setdefault("WORLD_SIZE", "1")
    
    # Configure Accelerator to avoid multi-node issues if MASTER_ADDR is not set
    # When accelerate launch is used, it may detect distributed mode but MASTER_ADDR might not be set
    # Set it to localhost for single-node multi-GPU setups
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    # Only set MASTER_PORT if not already set - let accelerate handle port selection if possible
    # If we must set it, use a random port to avoid conflicts
    if "MASTER_PORT" not in os.environ:
        import socket
        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            available_port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(available_port)
    
    accelerator = Accelerator()
    device = accelerator.device

    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    model = model.to(device)
    
    # Enable memory optimizations to prevent OOM
    if accelerator.is_main_process:
        print("Enabling VAE memory optimizations")
    model.enable_vae_slicing()  # Process VAE in slices to reduce memory
    try:
        model.enable_vae_tiling()  # Enable VAE tiling for large images
        if accelerator.is_main_process:
            print("VAE tiling enabled")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"VAE tiling not available: {e}")
    
    print("model loaded")
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
    print(f"Loading SAE from: {sae_checkpoint}")
    sae = load_sae(sae_checkpoint, hookpoint, device)
    with open(
        style_latents_path,
        "rb",
    ) as f:
        style_latents_dict = pickle.load(f)

    # Determine which themes to unlearn
    all_themes = [t for t in theme_available if t != "Seed_Images"]
    
    if themes_to_unlearn is None:
        # Default: unlearn all themes
        theme_avail = all_themes
        if accelerator.is_main_process:
            print(f"Unlearning all {len(theme_avail)} themes")
    else:
        # Validate and filter themes
        if isinstance(themes_to_unlearn, str):
            # Handle comma-separated string or single theme
            themes_to_unlearn = [t.strip() for t in themes_to_unlearn.split(",") if t.strip()]
        
        # Validate themes exist
        invalid_themes = [t for t in themes_to_unlearn if t not in theme_available]
        if invalid_themes:
            raise ValueError(
                f"Invalid themes specified: {invalid_themes}. "
                f"Available themes: {theme_available}"
            )
        
        # Filter out "Seed_Images" if included and get valid themes
        theme_avail = [t for t in themes_to_unlearn if t != "Seed_Images"]
        
        if not theme_avail:
            raise ValueError(
                "No valid themes to unlearn after filtering. "
                "Note: 'Seed_Images' is excluded from unlearning."
            )
        
        if accelerator.is_main_process:
            print(f"Unlearning {len(theme_avail)} specified theme(s): {theme_avail}")
    
    progress_bar = tqdm(
        theme_avail, total=len(theme_avail), disable=not accelerator.is_main_process
    )
    if accelerator.is_main_process:
        print(f"Themes to unlearn: {theme_avail}")
    print("progress bar")
    
    # Timing statistics
    concept_times = []
    inference_times = []
    io_times = []
    
    for theme_to_unlearn in progress_bar:
        concept_start_time = time.time()
        if accelerator.is_main_process:
            progress_bar.set_description(f"Unlearning {theme_to_unlearn}")
            print(f"\n[Timing] Starting concept: {theme_to_unlearn}")
        output_path = os.path.join(
            output_dir,
            f"percentile_{percentile}_multiplier_{multiplier}/{theme_to_unlearn}",
        )
        os.makedirs(output_path, exist_ok=True)
        
        test_theme_times = []
        for test_theme in theme_avail:
            test_theme_start = time.time()
            input_classes = []
            input_themes = []
            class_theme_pairs = [(c, test_theme) for c in class_available] + [
                (c, "") for c in class_available
            ]
            with accelerator.split_between_processes(
                class_theme_pairs
            ) as local_classes_themes:
                local_prompts = []
                local_prompt_metadata = []  # Store (object_class, theme) for each prompt
                for object_class, theme in local_classes_themes:
                    if theme == "":
                        local_prompts.append(f"An image of {object_class}.")
                    else:
                        local_prompts.append(
                            f"An image of {object_class} in {theme.replace('_', ' ')} style."
                        )
                    local_prompt_metadata.append((object_class, theme))
                
                # Process in batches if batch_size is specified
                if batch_size is None or batch_size <= 0:
                    # Process all at once (original behavior)
                    effective_batch_size = len(local_prompts)
                else:
                    effective_batch_size = batch_size
                    if accelerator.is_main_process and len(local_prompts) > effective_batch_size:
                        num_batches = (len(local_prompts) + effective_batch_size - 1) // effective_batch_size
                        print(f"  [Batching] Processing {len(local_prompts)} prompts in {num_batches} batches of {effective_batch_size}")
                
                all_images = []
                all_metadata = []
                
                # Process prompts in batches
                for batch_idx in range(0, len(local_prompts), effective_batch_size):
                    batch_prompts = local_prompts[batch_idx:batch_idx + effective_batch_size]
                    batch_metadata = local_prompt_metadata[batch_idx:batch_idx + effective_batch_size]
                    
                    steering_hooks = {}
                    steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                        concept_to_unlearn=[theme_to_unlearn],
                        percentile=percentile,
                        multiplier=multiplier,
                        feature_importance_fn=compute_feature_importance,
                        concept_latents_dict=style_latents_dict,
                        sae=sae,
                        steps=steps,
                        preserve_error=True,
                    )
                    with torch.no_grad():
                        inference_start = time.time()
                        batch_images = model.run_with_hooks(
                            prompt=batch_prompts,
                            generator=generator,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            position_hook_dict=steering_hooks,
                        )
                        inference_time = time.time() - inference_start
                        
                        # Handle single image vs list
                        if not isinstance(batch_images, list):
                            batch_images = [batch_images]
                        
                        all_images.extend(batch_images)
                        all_metadata.extend(batch_metadata)
                        
                        if accelerator.is_main_process:
                            inference_times.append(inference_time)
                            batch_num = batch_idx // effective_batch_size + 1
                            total_batches = (len(local_prompts) + effective_batch_size - 1) // effective_batch_size
                            print(f"  [Timing] Batch {batch_num}/{total_batches}: {inference_time:.2f}s for {len(batch_prompts)} prompts "
                                  f"({inference_time/len(batch_prompts):.2f}s per image)")
                        
                        # Clear cache to free up memory after each batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Store images and metadata for gathering
                images = all_images
                for object_class, theme in all_metadata:
                    input_classes.extend([object_class])
                    input_themes.extend([theme])
            accelerator.wait_for_everyone()
            images = gather_object(images)
            input_classes = gather_object(input_classes)
            input_themes = gather_object(input_themes)
            if accelerator.is_main_process:
                io_start = time.time()
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
                io_time = time.time() - io_start
                io_times.append(io_time)
                test_theme_time = time.time() - test_theme_start
                test_theme_times.append(test_theme_time)
                print(f"  [Timing] I/O: {io_time:.2f}s, Total test_theme: {test_theme_time:.2f}s")
        
        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            concept_time = time.time() - concept_start_time
            concept_times.append(concept_time)
            avg_test_theme_time = sum(test_theme_times) / len(test_theme_times) if test_theme_times else 0
            
            # Calculate inference stats for this concept
            num_batches_this_concept = len(test_theme_times)
            if inference_times and len(inference_times) >= num_batches_this_concept:
                concept_inference_times = inference_times[-num_batches_this_concept:]
                avg_inference = sum(concept_inference_times) / len(concept_inference_times)
                total_inference_time = sum(concept_inference_times)
            else:
                avg_inference = 0
                total_inference_time = 0
            
            print(f"[Timing] Concept '{theme_to_unlearn}' completed:")
            print(f"  Total time: {concept_time:.2f}s ({concept_time/60:.2f} minutes)")
            print(f"  Avg per test_theme: {avg_test_theme_time:.2f}s")
            print(f"  Total test_themes: {len(test_theme_times)}")
            print(f"  Avg inference per batch: {avg_inference:.2f}s")
            print(f"  Total inference time: {total_inference_time:.2f}s ({total_inference_time/concept_time*100:.1f}% of total)")
            print("-" * 60)
    
    # Final summary
    if accelerator.is_main_process and concept_times:
        print("\n" + "=" * 60)
        print("[Timing] FINAL SUMMARY")
        print("=" * 60)
        total_time = sum(concept_times)
        avg_concept_time = sum(concept_times) / len(concept_times)
        min_concept_time = min(concept_times)
        max_concept_time = max(concept_times)
        
        print(f"Total concepts processed: {len(concept_times)}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes, {total_time/3600:.2f} hours)")
        print(f"Average per concept: {avg_concept_time:.2f}s ({avg_concept_time/60:.2f} minutes)")
        print(f"Min concept time: {min_concept_time:.2f}s ({min_concept_time/60:.2f} minutes)")
        print(f"Max concept time: {max_concept_time:.2f}s ({max_concept_time/60:.2f} minutes)")
        
        if inference_times:
            total_inference = sum(inference_times)
            avg_inference = sum(inference_times) / len(inference_times)
            print(f"\nTotal inference calls: {len(inference_times)}")
            print(f"Total inference time: {total_inference:.2f}s ({total_inference/60:.2f} minutes)")
            print(f"Average inference per batch: {avg_inference:.2f}s")
            print(f"Inference time percentage: {total_inference/total_time*100:.1f}%")
        
        if io_times:
            total_io = sum(io_times)
            avg_io = sum(io_times) / len(io_times)
            print(f"\nTotal I/O operations: {len(io_times)}")
            print(f"Total I/O time: {total_io:.2f}s ({total_io/60:.2f} minutes)")
            print(f"Average I/O per batch: {avg_io:.2f}s")
            print(f"I/O time percentage: {total_io/total_time*100:.1f}%")
        
        print("=" * 60)


if __name__ == "__main__":
    fire.Fire(main)
