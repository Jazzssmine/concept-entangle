import argparse, os, random, json, copy
from typing import List, Dict

import torch
import inspect
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.loaders import AttnProcsLayers

# ---- import your graph utils ----
from graph_utils import load_superclass_graph, load_edge_list_graph

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def node_to_prompt(node_id: str, dataset: str) -> str:
    if dataset == "cifar":
        if node_id.startswith("fine_"):
            name = node_id[len("fine_"):]
            return f"a photo of a {name}"
        if node_id.startswith("coarse_"):
            name = node_id[len("coarse_"):]
            return f"a photo of a {name}"
    elif dataset == "style":
        if node_id.startswith("style_"):
            name = node_id[len("style_"):].replace("_", " ")
            return f"in the style of {name}"
    return node_id

class LoraLayersWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def parameters(self, recurse: bool = True):
        return (p for p in self.unet.parameters() if p.requires_grad)

    def save_pretrained(self, save_directory: str):
        # Uses diffusers' built-in LoRA save (PEFT backend).
        self.unet.save_attn_procs(save_directory)


def _inject_peft_lora(unet, rank: int = 8):
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA training with this diffusers version. "
            "Install it with: pip install peft"
        ) from exc

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    unet.add_adapter(lora_config, adapter_name="default")
    unet.set_adapter("default")
    return LoraLayersWrapper(unet)


def inject_lora_unet(unet, rank: int = 8):
    # If LoRA processors aren't Modules in this diffusers build, use PEFT.
    if not issubclass(LoRAAttnProcessor, torch.nn.Module):
        return _inject_peft_lora(unet, rank=rank)

    lora_attn_procs = {}
    for name, attn_proc in unet.attn_processors.items():
        if hasattr(attn_proc, "hidden_size"):
            hidden_size = attn_proc.hidden_size
        else:
            # Fallback for AttnProcessor2_0: derive hidden size from module name
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = unet.config.block_out_channels[-1]
        cross_attention_dim = getattr(attn_proc, "cross_attention_dim", None)
        init_sig = inspect.signature(LoRAAttnProcessor.__init__)
        kwargs = {}
        if "hidden_size" in init_sig.parameters:
            kwargs["hidden_size"] = hidden_size
        if "cross_attention_dim" in init_sig.parameters:
            kwargs["cross_attention_dim"] = cross_attention_dim
        if "rank" in init_sig.parameters:
            kwargs["rank"] = rank
        if "lora_rank" in init_sig.parameters:
            kwargs["lora_rank"] = rank
        lora_attn_procs[name] = LoRAAttnProcessor(**kwargs)
    unet.set_attn_processor(lora_attn_procs)
    return AttnProcsLayers(unet.attn_processors)

@torch.no_grad()
def encode_prompts(pipe, prompts: List[str], device: str):
    tok = pipe.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    return pipe.text_encoder(**tok).last_hidden_state

def make_forget_prompts(concept: str, n: int = 128) -> List[str]:
    templates = [
        "a photo of a {c}",
        "a picture of a {c}",
        "an image of a {c}",
        "a {c} on a table",
        "a close-up photo of a {c}",
    ]
    return [templates[i % len(templates)].format(c=concept) for i in range(n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--dataset", type=str, choices=["cifar", "style"], required=True)
    ap.add_argument("--graph_path", type=str, default="./data/cifar100/classes.json")  # superclasses.json or style_graph.json
    ap.add_argument("--target", type=str, default="bicycle")      # e.g. "bicycle" (cifar) or "Pencil Drawing" (style)
    ap.add_argument("--out_dir", type=str, default="outputs/cifar100/unlearn")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lambda_forget", type=float, default=1.0)
    ap.add_argument("--lambda_retain", type=float, default=0.0)
    ap.add_argument("--pi_mode", type=str, default="fixed", choices=["fixed", "custom"])  # fixed or user-provided
    ap.add_argument(
        "--pi_values",
        type=str,
        default="",
        help=(
            "Used when --pi_mode custom. Either JSON dict mapping anchor->prob "
            "(e.g. '{\"fine_bus\":0.4,\"fine_train\":0.6}') or comma list aligned to anchors "
            "(e.g. '0.4,0.3,0.2,0.1'). Values must be >= 0 and sum to 1."
        ),
    )
    ap.add_argument("--alpha_parent", type=float, default=0.0)  # set 0 for CIFAR peers-only
    ap.add_argument("--x_t_mode", type=str, default="random", choices=["random", "partial"])
    ap.add_argument("--t_min", type=int, default=200)
    ap.add_argument("--t_max", type=int, default=800)
    ap.add_argument("--t_start", type=int, default=560)
    ap.add_argument("--t_target", type=int, default=530)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp16", "bf16", "no"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)

    # ---- load graph ----
    if args.dataset == "cifar":
        G = load_superclass_graph(args.graph_path)  # builds coarse_/fine_ ids
        u = f"fine_{args.target}"
    else:
        G = load_edge_list_graph(args.graph_path)
        u = "style_" + args.target.replace(" ", "_") if not args.target.startswith("style_") else args.target

    anchors = G.anchor_set(u)

    # CIFAR: peers-only recommended
    if args.dataset == "cifar":
        anchors = [a for a in anchors if a.startswith("fine_")]

    if len(anchors) == 0:
        raise ValueError(f"No anchors found for {u}. Check graph or target name.")

    # fixed pi (uniform over anchors, or parent+peers if you later allow parent)
    if args.pi_mode == "fixed":
        if args.alpha_parent > 0:
            pi_dict = G.compute_pi(u, anchors, mode="fixed", alpha=args.alpha_parent)
        else:
            # uniform over anchors
            pi_dict = {a: 1.0 / len(anchors) for a in anchors}
    elif args.pi_mode == "custom":
        if not args.pi_values.strip():
            raise ValueError("--pi_values is required when --pi_mode custom.")

        raw = args.pi_values.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON for --pi_values: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("When JSON is used for --pi_values, it must be an object mapping anchor->prob.")
            missing = [a for a in anchors if a not in parsed]
            extra = [k for k in parsed.keys() if k not in anchors]
            if missing or extra:
                raise ValueError(f"--pi_values keys must exactly match anchors. Missing={missing}, extra={extra}")
            pi_dict = {a: float(parsed[a]) for a in anchors}
        else:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            if len(parts) != len(anchors):
                raise ValueError(
                    f"Comma --pi_values must have exactly {len(anchors)} entries (same order as anchors), got {len(parts)}."
                )
            vals = [float(p) for p in parts]
            pi_dict = {a: v for a, v in zip(anchors, vals)}

        if any(v < 0 for v in pi_dict.values()):
            raise ValueError(f"--pi_values must be non-negative. Got: {pi_dict}")
        s = sum(pi_dict.values())
        if s <= 0:
            raise ValueError(f"--pi_values must sum to a positive value. Got sum={s}")
        # Normalize to be robust to tiny floating-point mismatch.
        pi_dict = {a: (v / s) for a, v in pi_dict.items()}
    else:
        raise ValueError(f"Unknown pi_mode: {args.pi_mode}")

    anchor_prompts = [node_to_prompt(a, args.dataset) for a in anchors]
    pi = torch.tensor([pi_dict[a] for a in anchors], device=device, dtype=torch.float32)

    # forget prompts
    if args.dataset == "cifar":
        forget_prompts = make_forget_prompts(args.target, n=256)
    else:
        # style forget prompt
        style_name = args.target if not args.target.startswith("style_") else args.target[len("style_"):].replace("_", " ")
        forget_prompts = [f"in the style of {style_name}"] * 256

    print("Target node:", u)
    print("Anchors:", anchors)
    print("Anchor prompts:", anchor_prompts)
    print("Pi:", {a: float(pi_dict[a]) for a in anchors})

    # ---- load SD ----
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # teacher = frozen base UNet snapshot (must stay LoRA-free)
    teacher_unet = copy.deepcopy(pipe.unet).to(device)
    teacher_unet.requires_grad_(False)
    teacher_unet.eval()

    # student = same UNet but with LoRA enabled (trainable)
    lora_layers = inject_lora_unet(pipe.unet, rank=args.rank)
    lora_layers.to(device).train()

    opt = torch.optim.AdamW(lora_layers.parameters(), lr=args.lr)
    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    use_amp = args.mixed_precision != "no"
    # GradScaler can't be used with fp16 parameters; disable when model is fp16.
    use_scaler = args.mixed_precision == "fp16" and device == "cuda" # and dtype != torch.float16
    scaler = GradScaler(amp_device, enabled=use_scaler)

    # training noise scheduler
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    n_train_ts = int(noise_scheduler.config.num_train_timesteps)
    if not (0 <= args.t_min < args.t_max < n_train_ts):
        raise ValueError(
            f"Require 0 <= t_min < t_max < {n_train_ts}, got t_min={args.t_min}, t_max={args.t_max}"
        )
    if not (0 <= args.t_target < args.t_start < n_train_ts):
        raise ValueError(
            f"Require 0 <= t_target < t_start < {n_train_ts}, got t_target={args.t_target}, t_start={args.t_start}"
         )

    # pre-encode anchor embeddings (K,77,D)
    with torch.no_grad():
        anchor_emb = encode_prompts(pipe, anchor_prompts, device=device)

    # loop
    K = len(anchor_prompts)
    for step in tqdm(range(1, args.steps + 1)):
        # sample forget prompts
        fp = [forget_prompts[(step + i) % len(forget_prompts)] for i in range(args.batch)]
        # sample retain prompts from anchors (pi)
        idx = torch.multinomial(pi, num_samples=args.batch, replacement=True).tolist()
        rp = [anchor_prompts[j] for j in idx]

        with torch.no_grad():
            forget_emb = encode_prompts(pipe, fp, device=device)
            retain_emb = encode_prompts(pipe, rp, device=device)
            if args.x_t_mode == "partial":
                # Build x_t_target by partially denoising from pure noise under forget prompts.
                noisy_latents = torch.randn(args.batch, 4, 64, 64, device=device, dtype=dtype)
                for t in range(args.t_start, args.t_target, -1):
                    t_in = torch.full((args.batch,), t, device=device, dtype=torch.long)
                    eps_t = teacher_unet(noisy_latents, t_in, encoder_hidden_states=forget_emb).sample
                    noisy_latents = noise_scheduler.step(eps_t, t, noisy_latents).prev_sample
                timesteps = torch.full((args.batch,), args.t_target, device=device, dtype=torch.long)
            else:
                # Faster baseline: sample x_t from one random timestep.
                latents = torch.randn(args.batch, 4, 64, 64, device=device, dtype=dtype)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(args.t_min, args.t_max + 1, (args.batch,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        opt.zero_grad(set_to_none=True)

        with autocast(device_type=amp_device, enabled=use_amp, dtype=dtype):
            # student under forget prompts
            eps_s_forget = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=forget_emb).sample

            # mixture teacher target under anchor prompts
            with torch.no_grad():
                eps_mix = torch.zeros_like(eps_s_forget)
                for j in range(K):
                    emb_j = anchor_emb[j].unsqueeze(0).expand(args.batch, -1, -1)
                    eps_j = teacher_unet(noisy_latents, timesteps, encoder_hidden_states=emb_j).sample
                    eps_mix = eps_mix + pi[j] * eps_j
            # import pdb; pdb.set_trace()
            loss_forget = F.mse_loss(eps_s_forget, eps_mix)

            # retain distillation
            eps_s_retain = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=retain_emb).sample
            with torch.no_grad():
                eps_t_retain = teacher_unet(noisy_latents, timesteps, encoder_hidden_states=retain_emb).sample
            loss_retain = F.mse_loss(eps_s_retain, eps_t_retain)

            loss = args.lambda_forget * loss_forget + args.lambda_retain * loss_retain

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        if step % 100 == 0:
            print(
                f"step={step} loss={loss.item():.6f} forget={loss_forget.item():.6f} "
                f"retain={loss_retain.item():.6f} t_mean={timesteps.float().mean().item():.1f}"
            )

        if step % 500 == 0:
            save_path = os.path.join(args.out_dir, f"lora_step{step}")
            os.makedirs(save_path, exist_ok=True)
            lora_layers.save_pretrained(save_path)
            print("Saved LoRA:", save_path)

    save_path = os.path.join(args.out_dir, "lora_final")
    os.makedirs(save_path, exist_ok=True)
    lora_layers.save_pretrained(save_path)
    print("Saved LoRA:", save_path)

if __name__ == "__main__":
    main()
