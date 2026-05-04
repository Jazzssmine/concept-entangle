# concept entanglement

Unified, cleaned README for the `concept entanglement` code release.

This repository focuses on diffusion-model concept entanglement evaluation and analysis:
- concept-set and prompt construction
- benchmark generation and CLIP-based evaluation
- activation-space experiments (Experiments 1-3)
- kappa and Pareto analysis utilities

Large artifacts (model weights, checkpoints, cached activations, generated images) are intentionally excluded from this cleaned release.

## Repository Layout

- `scripts/`: experiment and pipeline entrypoints
- `src/`: benchmark/runtime utilities
- `configs/`: model registries and benchmark maps
- `baselines/`: baseline-specific integration code
- `utils/`: helper utilities

## Environment

Install base dependencies:

```bash
pip install -r requirements.txt
```

For reproducibility, an environment spec is also provided:

```bash
conda env create -f environment.yml
```

## Quick Start

### 1) Build prompts

```bash
python scripts/build_prompts.py \
  --concept_sets_dir outputs/concept_sets/per_target \
  --output_dir outputs/prompts \
  --direct_per_target 200 \
  --indirect_per_target 200 \
  --neighbor_per_concept 20 \
  --control_per_concept 20 \
  --random_seed 42
```

### 2) Run benchmark generation

```bash
python scripts/run_benchmark.py \
  --prompts_path outputs/prompts/horse_prompts.csv \
  --model_registry_path configs/benchmark_models.example.json \
  --model_key base_horse \
  --output_dir outputs/benchmark \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --generation_only \
  --seeds 0
```

### 3) Evaluate generated images

```bash
python scripts/evaluate_from_image_folders.py \
  --images_root outputs/benchmark/images \
  --output_dir outputs/benchmark/eval \
  --model_key base_horse \
  --concept_vocab_path data/concept_vocab_objects.txt \
  --prompts_path outputs/prompts/horse_prompts.csv \
  --base_model_name base_horse \
  --clip_model_name openai/clip-vit-base-patch32 \
  --compute_cp \
  --save_topk
```

## Experiment Scripts

### Experiment 1: Activation Geometry

- Extraction: `scripts/extract_activations.py`
- Visualization: `scripts/visualize_activations.py`
- Main outputs: centroid distance plots, UMAP plots, `summary_v3.csv`

### Experiment 2: Lipschitz Validation

- Perturbation version: `scripts/run_experiment2_lipschitz.py`
- Prompt-variation version: `scripts/run_experiment2_lipschitz_prompt_variation.py`
- Main outputs: Lipschitz figures/tables and raw response CSVs

### Experiment 3: Activation Displacement

- Pipeline: `scripts/run_experiment3_displacement.py`
- Additional theorem-focused variant: `scripts/measure_activation_displacement.py`
- Main outputs: per-method displacement stats and summary figures

## Kappa Pipeline

- Main script: `scripts/run_kappa_updated_prompts.py`
- Batch helper: `scripts/run_kappa_updated_batches.sh`
- Typical outputs:
  - `kappa_table.csv`
  - concept-wise centroid distance plots
  - run logs and cache-backed intermediate files
