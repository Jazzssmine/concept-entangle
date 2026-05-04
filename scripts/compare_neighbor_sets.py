import argparse
import json
from pathlib import Path


def _load_neighbors(output_dir: Path) -> dict[str, set[str]]:
    out = {}
    for fp in (output_dir / "per_target").glob("*.json"):
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        out[obj["target"]] = set(obj["semantic_neighbors"]["final_top_k"])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare neighbor sets across two pipeline outputs.")
    parser.add_argument("--output_a", type=str, required=True)
    parser.add_argument("--output_b", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    a = _load_neighbors(Path(args.output_a))
    b = _load_neighbors(Path(args.output_b))
    targets = sorted(set(a.keys()) & set(b.keys()))
    if not targets:
        print("No overlapping targets found.")
        return

    print("target,jaccard,overlap_count,only_in_a,only_in_b")
    for t in targets:
        sa, sb = a[t], b[t]
        inter = sa & sb
        union = sa | sb
        jac = len(inter) / max(1, len(union))
        only_a = "|".join(sorted(sa - sb))
        only_b = "|".join(sorted(sb - sa))
        print(f"{t},{jac:.4f},{len(inter)},{only_a},{only_b}")


if __name__ == "__main__":
    main()

