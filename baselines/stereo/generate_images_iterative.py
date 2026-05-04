from diffusers import StableDiffusionPipeline
import torch
import os
import argparse
import re
import json
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative loop: prompt -> image generation -> OCR text extraction -> next prompt"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory", required=True)
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_checkpoint", type=str, help="Path to erased unet checkpoint", default="")
    parser.add_argument("--prompt", type=str, help="Single prompt for image generation", default="")
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Optional path to a file containing prompts (one per line or bucket|prompt format)",
        default="",
    )
    parser.add_argument("--num_images", type=int, help="Images to generate for each prompt in each round", default=2)
    parser.add_argument("--seed_offset", type=int, help="Seed offset to vary random generations across runs", default=0)
    parser.add_argument("--num_inference_steps", help="num_inference_steps", type=int, required=False, default=50)
    parser.add_argument("--guidance_scale", help="guidance_scale", type=float, required=False, default=7.5)
    parser.add_argument("--ocr_model", type=str, default="microsoft/trocr-base-printed", help="OCR model id")
    parser.add_argument("--ocr_iterations", type=int, default=3, help="Number of OCR loop rounds")
    parser.add_argument("--max_prompts_next_round", type=int, default=50, help="Maximum prompts to keep for next OCR round")
    parser.add_argument("--min_extracted_chars", type=int, default=3, help="Minimum OCR text length to be used as next prompt")
    parser.add_argument(
        "--keep_prompt_if_empty",
        action="store_true",
        help="Reuse source prompt if OCR extraction is empty/too short",
    )
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
            if "|" in line:
                _, prompt = line.split("|", 1)
                prompt = prompt.strip()
                if prompt:
                    prompts.append(prompt)
            else:
                prompts.append(line)
    return prompts


def unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def clean_ocr_text(text):
    return " ".join(text.split()).strip()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device)

    if not args.prompt and not args.prompt_file:
        raise ValueError("Provide either --prompt or --prompt_file.")
    if args.prompt and args.prompt_file:
        raise ValueError("Use only one of --prompt or --prompt_file.")

    if args.prompt_file:
        prompts = load_prompts(args.prompt_file)
        if len(prompts) == 0:
            raise ValueError(f"No prompts found in file: {args.prompt_file}")
    else:
        prompts = [args.prompt]

    os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, safety_checker=None, torch_dtype=torch_dtype
    ).to(device)

    if args.unet_checkpoint != "":
        print("Loading erased unet checkpoint from", args.unet_checkpoint)
        pipe.unet.load_state_dict(torch.load(args.unet_checkpoint, map_location=device))

    ocr_processor = TrOCRProcessor.from_pretrained(args.ocr_model)
    ocr_model = VisionEncoderDecoderModel.from_pretrained(args.ocr_model).to(device)
    ocr_model.eval()

    run_summary = {
        "model_path": args.model_path,
        "unet_checkpoint": args.unet_checkpoint,
        "ocr_model": args.ocr_model,
        "ocr_iterations": args.ocr_iterations,
        "num_images_per_prompt": args.num_images,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed_offset": args.seed_offset,
        "rounds": [],
    }

    with torch.no_grad():
        for round_idx in range(args.ocr_iterations):
            round_dir = os.path.join(args.output_dir, f"round_{round_idx}")
            images_dir = os.path.join(round_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            extracted_prompts = []
            records = []

            print(f"Round {round_idx + 1}/{args.ocr_iterations}: input prompts = {len(prompts)}")
            for prompt_idx, prompt in enumerate(prompts):
                filename = sanitize_filename(prompt)
                for image_idx in range(args.num_images):
                    seed = args.seed_offset + (round_idx * 100000) + (prompt_idx * 1000) + image_idx
                    gen.manual_seed(seed)
                    torch.manual_seed(seed)
                    out = pipe(
                        prompt=[prompt],
                        generator=gen,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                    )
                    image = out.images[0]
                    image_name = f"p{prompt_idx:03d}_i{image_idx:02d}_{filename}.png"
                    image_path = os.path.join(images_dir, image_name)
                    image.save(image_path)

                    pil_image = Image.open(image_path).convert("RGB")
                    pixel_values = ocr_processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
                    generated_ids = ocr_model.generate(pixel_values)
                    extracted_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    extracted_text = clean_ocr_text(extracted_text)

                    if len(extracted_text) >= args.min_extracted_chars:
                        extracted_prompts.append(extracted_text)
                    elif args.keep_prompt_if_empty:
                        extracted_prompts.append(prompt)

                    records.append(
                        {
                            "source_prompt": prompt,
                            "prompt_index": prompt_idx,
                            "image_index": image_idx,
                            "seed": seed,
                            "image_path": image_path,
                            "extracted_text": extracted_text,
                        }
                    )

            next_prompts = unique_keep_order(extracted_prompts)
            if args.max_prompts_next_round > 0:
                next_prompts = next_prompts[: args.max_prompts_next_round]

            round_metadata = {
                "round_index": round_idx,
                "input_prompts": prompts,
                "num_generated_images": len(records),
                "records": records,
                "next_prompts": next_prompts,
            }
            with open(os.path.join(round_dir, "round_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(round_metadata, f, indent=2, ensure_ascii=True)

            run_summary["rounds"].append(
                {
                    "round_index": round_idx,
                    "num_input_prompts": len(prompts),
                    "num_generated_images": len(records),
                    "num_next_prompts": len(next_prompts),
                    "metadata_path": os.path.join(round_dir, "round_metadata.json"),
                }
            )

            if len(next_prompts) == 0:
                print("No prompts extracted for next round, stopping early.")
                break
            prompts = next_prompts

    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=True)

    print(f"Done. Summary saved to {os.path.join(args.output_dir, 'run_summary.json')}")
