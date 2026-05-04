import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print concise concept-set summaries.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory from build_concept_sets.py")
    parser.add_argument("--top_n", type=int, default=10, help="How many items to print from each set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_target_dir = Path(args.output_dir) / "per_target"
    if not per_target_dir.exists():
        raise FileNotFoundError(f"Missing path: {per_target_dir}")

    for path in sorted(per_target_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        target = obj["target"]
        context = obj["context_words"]["final_top_k"][: args.top_n]
        neighbors = obj["semantic_neighbors"]["final_top_k"][: args.top_n]
        controls = obj["non_neighbor_controls"]["final_top_k"][: args.top_n]

        print("=" * 80)
        print(f"Target: {target}")
        print(f"Strict lexical: {obj['lexical_set']['strict']}")
        print(f"Broad lexical:  {obj['lexical_set']['broad']}")
        print(f"Context top-{args.top_n}:   {context}")
        print(f"Neighbors top-{args.top_n}: {neighbors}")
        print(f"Controls top-{args.top_n}:  {controls}")
    print("=" * 80)


if __name__ == "__main__":
    main()

