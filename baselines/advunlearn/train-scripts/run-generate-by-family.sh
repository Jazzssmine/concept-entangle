#!/usr/bin/env bash
set -euo pipefail

# One-shot image generation for benchmark prompt families:
# control, direct, indirect, neighbor
#
# Usage:
#   bash train-scripts/run-generate-by-family.sh \
#     --ckpt-base "<path-without-.pt>" \
#     --prompts-dir "/u/anon3/unlearn_diff/outputs/prompts/horse/for_generate_example" \
#     --device "cuda:0"
# bash train-scripts/run-generate-by-family.sh \
#   --ckpt-base "./results/results_with_retaining/horse/coco_object/fast_at/AttackLr_0.001/text_encoder_full/all/prefix_k/AdvUnlearn-horse-method_text_encoder_full_all-Attack_fast_at-Retain_coco_object_reg_0.3-lr_1e-05-AttackLr_0.001-prefix_k_adv_num_1-word_embd-attack_init_random-attack_step_5-adv_update_1-warmup_iter_200/models/TextEncoder-text_encoder_full-epoch_399" \
#   --prompts-dir "/u/anon3/unlearn_diff/outputs/prompts/horse/for_generate_example" \
#   --device "cuda:0" \
#   --num-samples 1 \
#   --ddim-steps 50

CKPT_BASE=""
PROMPTS_DIR=""
MODEL_NAME="SD-v1-4"
DEVICE="cuda:0"
NUM_SAMPLES="1"
DDIM_STEPS="50"
ORIGIN_OR_TARGET="target"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-base)
      CKPT_BASE="$2"
      shift 2
      ;;
    --prompts-dir)
      PROMPTS_DIR="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --ddim-steps)
      DDIM_STEPS="$2"
      shift 2
      ;;
    --origin-or-target)
      ORIGIN_OR_TARGET="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CKPT_BASE" ]]; then
  echo "Missing required flag: --ckpt-base" >&2
  exit 1
fi

if [[ -z "$PROMPTS_DIR" ]]; then
  echo "Missing required flag: --prompts-dir" >&2
  exit 1
fi

for FAM in control direct indirect neighbor; do
  CSV_PATH="${PROMPTS_DIR}/${FAM}.csv"
  if [[ ! -f "$CSV_PATH" ]]; then
    echo "Missing prompts CSV: $CSV_PATH" >&2
    exit 1
  fi

  echo "[run] family=${FAM}"
  python train-scripts/generate-example-img.py \
    --model_name "$MODEL_NAME" \
    --prompts_path "$CSV_PATH" \
    --save_path "$CKPT_BASE" \
    --folder_suffix "horse_${FAM}" \
    --origin_or_target "$ORIGIN_OR_TARGET" \
    --device "$DEVICE" \
    --num_samples "$NUM_SAMPLES" \
    --ddim_steps "$DDIM_STEPS"
done

echo "[done] Generated all prompt families."
