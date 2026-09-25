"""Build a 4x4 puzzle-shuffled MM-SafetyBench subset (used in the rebuttal experiments).

Loads an MM-SafetyBench split saved with the HuggingFace ``datasets`` library,
applies the Module-1 grid shuffle to each image, and writes the shuffled images
plus a metadata JSON. See ``DATA.md`` for how to obtain MM-SafetyBench.
"""

import argparse
import json
import os
import sys

from datasets import load_from_disk
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gambit import encoding


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", required=True, help="load_from_disk path.")
    ap.add_argument("--subset", default="SD", help="Split name (e.g. SD, SD_TYPO, TYPO).")
    ap.add_argument("--grid", type=int, default=4, help="Puzzle grid size N (default 4).")
    ap.add_argument("--out", default="mm_safetybench_puzzle", help="Output directory.")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_path)
    if args.subset not in ds:
        raise SystemExit(f"Subset {args.subset!r} not found. Available: {list(ds.keys())}")
    subset = ds[args.subset]

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    metadata = []
    for i, ex in enumerate(tqdm(subset, desc=args.subset)):
        try:
            puzzle = encoding.shuffle_image(ex["image"].convert("RGB"), grid=args.grid)
            ex_id = ex.get("id", str(i))
            path = os.path.join(img_dir, f"{ex_id}.jpg")
            puzzle.save(path)
            metadata.append({"id": ex_id, "question": ex.get("question", ""),
                             "image_path": path})
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping example {i}: {exc}")

    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(metadata)} shuffled items -> {args.out}")


if __name__ == "__main__":
    main()
