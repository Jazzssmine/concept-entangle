"""
Script for hyperparameter sweep for object unlearning.
"""
import sys
sys.stdout.flush()

import os
import pickle
from pathlib import Path

# Add parent directory to path so we can import utils and SAE
script_dir = Path(__file__).parent
saeuron_dir = script_dir.parent
sys.path.insert(0, str(saeuron_dir))

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object as _gather_object
from packaging import version
from tqdm import tqdm

import utils.hooks as hooks
from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from SAE.unlearning_utils import compute_feature_importance

import fire

from UnlearnCanvas_resources.const import class_available, theme_available

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
    target_class=None,
    no_styles=False,
    seed=42,
    steps=100,
    percentiles=[99.99, 99.995, 99.999],
    multipliers=[-1.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0],
    guidance_scale=9.0,
    output_dir="sweep_results/mu_results/class20/",
    limit_themes=50,
):
    # Print immediately to verify script is running
    print("=" * 50)
    print("SCRIPT STARTED - main() function called")
    print("=" * 50)
    import sys
    sys.stdout.flush()
    
    # Configure Accelerator to avoid multi-node issues if MASTER_ADDR is not set
    # When accelerate launch is used, it may detect distributed mode but MASTER_ADDR might not be set
    # Set it to localhost for single-node multi-GPU setups
    num_gpus = torch.cuda.device_count()
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    
    print(f"Number of CUDA devices available: {num_gpus}")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    
    # Check if we're actually in distributed mode
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Warn if multiple GPUs are available but not being used
    if num_gpus > 1 and world_size == 1:
        print(f"⚠️  WARNING: {num_gpus} GPUs are available but only 1 process is being used!")
        print(f"   To use all {num_gpus} GPUs, run: accelerate launch --num_processes {num_gpus} ...")
        print(f"   Or set CUDA_VISIBLE_DEVICES to limit which GPUs to use")
    
    # Only configure distributed settings if we have multiple processes/GPUs
    if world_size > 1 or num_gpus > 1:
        print(f"Distributed mode detected: WORLD_SIZE={world_size}, GPUs={num_gpus}")
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
            print("Set MASTER_ADDR to localhost")
        # Only set MASTER_PORT if not already set
        # IMPORTANT: All processes must use the SAME port for distributed training to work
        # Use a deterministic port based on output_dir so all processes agree
        if "MASTER_PORT" not in os.environ:
            import hashlib
            port_base = 29500
            # Generate a deterministic port based on output_dir to ensure all processes use the same port
            port_hash = int(hashlib.md5(output_dir.encode()).hexdigest()[:4], 16) % 1000
            master_port = port_base + port_hash
            os.environ["MASTER_PORT"] = str(master_port)
            print(f"Set MASTER_PORT to {master_port} (based on output_dir hash)")
    else:
        print(f"Single GPU mode: WORLD_SIZE={world_size}, GPUs={num_gpus}")
        # For single GPU, clear distributed env vars to avoid issues
        if "MASTER_ADDR" in os.environ and os.environ["MASTER_ADDR"] == "localhost":
            # Keep it but it won't be used
            pass
    
    sys.stdout.flush()  # Force flush to see output immediately
    
    # For single GPU, we can skip Accelerator to avoid distributed setup overhead
    # But if accelerate launch is used, we need Accelerator for proper coordination
    use_accelerator = world_size > 1 or "RANK" in os.environ or "LOCAL_RANK" in os.environ
    
    if use_accelerator:
        print("Using Accelerator (distributed mode)")
        accelerator = Accelerator()
        print("======Accelerator initialized=======")
        device = accelerator.device
        print(f"Device: {device}, Process rank: {accelerator.process_index}, World size: {accelerator.num_processes}")
    else:
        print("Single GPU mode - skipping Accelerator")
        # Create a simple wrapper for single GPU
        class SingleGPUAccelerator:
            def __init__(self):
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.process_index = 0
                self.num_processes = 1
                self.is_main_process = True
            def wait_for_everyone(self):
                pass
            def split_between_processes(self, items):
                # Return a context manager that yields all items for single process
                class SingleProcessContext:
                    def __init__(self, items):
                        self.items = items
                    def __enter__(self):
                        return self.items
                    def __exit__(self, *args):
                        pass
                return SingleProcessContext(items)
        accelerator = SingleGPUAccelerator()
        device = accelerator.device
        print(f"Device: {device}")
    sys.stdout.flush()

    sys.stdout.flush()
    
    # Enable CPU offloading to save GPU memory
    # For single GPU, always enable CPU offloading to avoid OOM
    # This moves model components to CPU when not in use
    enable_cpu_offload = os.environ.get("ENABLE_CPU_OFFLOAD", "auto").lower()
    if enable_cpu_offload == "auto":
        # Auto-enable for single GPU to prevent OOM
        enable_cpu_offload = (num_gpus == 1 and world_size == 1)
    else:
        enable_cpu_offload = enable_cpu_offload == "true"
    
    model = HookedStableDiffusionPipeline.from_pretrained(
        pipe_checkpoint,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    sys.stdout.flush()
    
    if enable_cpu_offload:
        print("Enabling CPU offloading to save GPU memory (single GPU mode)")
        model.enable_model_cpu_offload()
        device = "cpu"  # CPU offloading uses CPU as main device
    else:
        print(f"Loading model to GPU: {device}")
        model = model.to(device)
        # Enable VAE CPU offloading even in multi-GPU mode to save memory
        # VAE decoding is very memory intensive
        print("Enabling VAE CPU offloading to reduce memory usage during decoding")
        model.enable_vae_slicing()  # Process VAE in slices to reduce memory
        # Also enable VAE tiling for large images
        try:
            model.enable_vae_tiling()
            print("VAE tiling enabled")
        except:
            print("VAE tiling not available (may need newer diffusers version)")
    
    print("======model loaded=======")
    sys.stdout.flush()

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
    print("======xformers enabled=======")
    seed_everything(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sae = load_sae(sae_checkpoint, hookpoint, device)
    print("======sae loaded=======")
    with open(
        class_latents_path,
        "rb",
    ) as f:
        class_latents_dict = pickle.load(f)
    print("======class latents dict loaded=======")
    if target_class is not None:
        if target_class not in class_available:
            raise ValueError(
                f"Invalid target_class '{target_class}'. Must be one of: {class_available}"
            )
        classes_to_unlearn = [target_class]
    else:
        classes_to_unlearn = class_available

    class_prompt_dict = {class_: [] for class_ in class_available}
    if no_styles:
        for class_name in class_available:
            class_prompt_dict[class_name].append(f"An image of {class_name}.")
    else:
        for class_to_unlearn in class_available:
            with open(
                os.path.join(
                    "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                    f"sd_prompt_{class_to_unlearn}.txt",
                ),
                "r",
            ) as prompt_file:
                prompts = prompt_file.readlines()
                for i, theme in enumerate(theme_available):
                    if i >= limit_themes:
                        break
                    if theme != "Seed_Images":
                        theme_prompt = prompts[i]
                        theme_prompt = theme_prompt.strip()
                        theme_prompt = (
                            theme_prompt
                            if not theme_prompt.endswith(".")
                            else theme_prompt[:-1]
                        )
                        theme_prompt = f"{theme_prompt} in {theme.replace('_', ' ')} style."
                        class_prompt_dict[class_to_unlearn].append(theme_prompt)

    progress_bar = tqdm(
        total=len(multipliers) * len(classes_to_unlearn) * len(percentiles),
        disable=not accelerator.is_main_process,
    )
    print("======progress bar set=======")
    for multiplier in multipliers:
        for percentile in percentiles:
            for class_to_unlearn in classes_to_unlearn:
                if accelerator.is_main_process:
                    progress_bar.set_description(
                        f"Multiplier: {multiplier} Percentile: {percentile} Class: {class_to_unlearn}"
                    )
                output_path = os.path.join(
                    output_dir,
                    f"percentile_{percentile}_multiplier_{multiplier}/{class_to_unlearn}",
                )
                os.makedirs(output_path, exist_ok=True)
                all_prompts = [
                    (class_name, prompt)
                    for class_name, prompts in class_prompt_dict.items()
                    for prompt in prompts
                ]
                input_classes = []
                with accelerator.split_between_processes(all_prompts) as local_tuples:
                    local_prompts = [prompt.strip() for _, prompt in local_tuples]
                    local_classes = [class_name for class_name, _ in local_tuples]
                    steering_hooks = {}
                    steering_hooks[hookpoint] = hooks.SAEMaskedUnlearningHook(
                        concept_to_unlearn=[class_to_unlearn],
                        percentile=percentile,
                        multiplier=multiplier,
                        feature_importance_fn=compute_feature_importance,
                        concept_latents_dict=class_latents_dict,
                        sae=sae,
                        steps=steps,
                        preserve_error=True,
                    )
                    
                    # Process images one at a time to reduce memory usage
                    # VAE decoding is very memory intensive
                    images = []
                    with torch.no_grad():
                        for i, prompt in enumerate(local_prompts):
                            # Clear cache before each image to free up memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Reset hook timestep_idx for each new image
                            # The hook's timestep_idx gets incremented during denoising
                            # and needs to be reset for each new image
                            steering_hooks[hookpoint].timestep_idx = 0
                            
                            # Process one image at a time
                            image = model.run_with_hooks(
                                prompt=[prompt],  # Single prompt
                                generator=generator,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                                position_hook_dict=steering_hooks,
                            )
                            
                            # Handle single image vs list
                            if isinstance(image, list):
                                images.extend(image)
                            else:
                                images.append(image)
                            
                            # Clear cache after each image
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    input_classes.extend(local_classes)
                accelerator.wait_for_everyone()
                # Use gather_object for distributed, or just return the list for single GPU
                if accelerator.num_processes > 1:
                    images = _gather_object(images)
                    input_classes = _gather_object(input_classes)
                else:
                    # Single GPU - no gathering needed
                    images = images if isinstance(images, list) else [images]
                    input_classes = input_classes if isinstance(input_classes, list) else [input_classes]
                if accelerator.is_main_process:
                    for i, (img, object_class) in enumerate(zip(images, input_classes)):
                        img.save(
                            os.path.join(
                                output_path,
                                f"{object_class}_seed{seed}_{i}.jpg",
                            )
                        )
                if accelerator.is_main_process:
                    progress_bar.update(1)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    print("=" * 50)
    print("ABOUT TO CALL fire.Fire(main)")
    print("=" * 50)
    sys.stdout.flush()
    fire.Fire(main)
    
