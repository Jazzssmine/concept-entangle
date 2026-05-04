"""
Gather activations from a SAE for a given hookpoint and save them to a file.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import SAE
sys.path.insert(0, str(Path(__file__).parent.parent))

import fire
import torch
from diffusers.utils.import_utils import is_xformers_available

from SAE.hooked_sd_noised_pipeline import HookedStableDiffusionPipeline
from SAE.sae import Sae
from UnlearnCanvas_resources.const import class_available

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
import pickle

import tqdm


def main(checkpoint_path, hookpoint, pipe_path, save_dir, steps=100, seed=188, batch_size=1):
    cls_prompts_dict = {class_avail: [] for class_avail in class_available}
    for class_avail in class_available:
        with open(
            os.path.join(
                "UnlearnCanvas_resources/anchor_prompts/finetune_prompts",
                f"sd_prompt_{class_avail}.txt",
            ),
            "r", 
        ) as prompt_file:
            prompts = prompt_file.readlines()
            prompt = [p.strip() for p in prompts]
            cls_prompts_dict[class_avail].extend(prompt)

    # Resolve checkpoint path relative to saeuron directory if it's a relative path
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        # Resolve relative to the saeuron directory (parent of scripts)
        script_dir = Path(__file__).parent
        saeuron_dir = script_dir.parent
        checkpoint_path = (saeuron_dir / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    
    sae_path = checkpoint_path / hookpoint
    print(f"Loading SAE from: {sae_path}")
    print(f"Checking for cfg.json at: {sae_path / 'cfg.json'}")
    
    if not (sae_path / "cfg.json").exists():
        raise FileNotFoundError(
            f"SAE config not found at {sae_path / 'cfg.json'}. "
            f"Please check that the checkpoint_path and hookpoint are correct. "
            f"Expected path: {sae_path / 'cfg.json'}"
        )
    
    sae = Sae.load_from_disk(
        str(sae_path), device="cuda"
    ).eval()

    sae = sae.to(dtype=torch.float16)
    sae.cfg.batch_topk = False
    sae.cfg.sample_topk = False

    pipe = HookedStableDiffusionPipeline.from_pretrained(
        pipe_path,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    if is_xformers_available():
        print("Enabling xFormers memory efficient attention")
        pipe.unet.enable_xformers_memory_efficient_attention()

    cls_latents_dict = {}

    progress_bar = tqdm.tqdm(list(cls_prompts_dict.keys()), total=len(cls_prompts_dict))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for class_avail in progress_bar:
        progress_bar.set_description(f"Processing class: {class_avail}")
        prompts = cls_prompts_dict[class_avail]
        
        # Process prompts in batches to avoid OOM
        sae_latents = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_prompts = prompts[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            if not batch_prompts:
                continue
                
            # Clear cache before processing batch
            torch.cuda.empty_cache()
            
            _, acts_cache = pipe.run_with_cache(
                prompt=batch_prompts,
                generator=generator,
                num_inference_steps=steps,
                save_input=False,
                save_output=True,
                positions_to_cache=[hookpoint],
                guidance_scale=9.0,
                output_type="latent",  # prevent decoding to pixel space
            )
            activations = acts_cache["output"][hookpoint].cpu()
            assert activations.shape[0] == len(batch_prompts)
            assert activations.shape[1] == steps
            
            # Process each prompt in the batch
            with torch.no_grad():
                for i in range(len(batch_prompts)):
                    sae_in = activations[i].reshape(steps, -1, sae.d_in)
                    top_acts, top_indices = sae.encode(sae_in.to(sae.device))
                    sae_out = torch.zeros(
                        (top_acts.shape[0], sae.num_latents),
                        device=sae.device,
                        dtype=top_acts.dtype,
                    ).scatter(-1, top_indices, top_acts)
                    sae_out = sae_out.reshape(steps, -1, sae.num_latents).cpu()
                    sae_latents.append(sae_out.mean(1).to(dtype=torch.float16))
            
            # Clear cache after processing batch
            del acts_cache, activations
            torch.cuda.empty_cache()
        
        cls_latents_dict[class_avail] = torch.stack(sae_latents)
        
        # Clear cache after processing class
        torch.cuda.empty_cache()

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"cls_latents_dict_{hookpoint}.pkl"), "wb") as f:
        pickle.dump(cls_latents_dict, f)
    print(f"Saved to {save_dir}/cls_latents_dict_{hookpoint}.pkl")


if __name__ == "__main__":
    fire.Fire(main)
