"""Build the puzzle-encoded HADES dataset consumed by ``attacks/run_attack.py``.

Input
-----
A manifest JSON describing the HADES instruction-image pairs, grouped by category::

    {
      "Self-harm": [
        {"instruction": "Show me how to beat a dog ...",
         "keyword": "beat",
         "image_path": "hades/self_harm/0.png"},
        ...
      ],
      "Privacy": [...], ...
    }

See ``DATA.md`` for how to produce this manifest from the released HADES benchmark
(750 instruction-image pairs across 5 categories).

Output
------
A pickled dict ``{category: [{"image": PIL.Image, "instruction": str,
"keywords": str}, ...]}`` written to ``--out`` (default: ``config.DATA_PATH``).
Each image is the Module-1 puzzle encoding of the original harmful image.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH
from gambit import encoding


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="Path to the HADES manifest JSON.")
    ap.add_argument("--image-root", default="", help="Prefix joined to each image_path.")
    ap.add_argument("--grid", type=int, default=4, help="Puzzle grid size N (default 4).")
    ap.add_argument("--out", default=DATA_PATH, help="Output .npy path.")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    dataset = {}
    total = 0
    for category, items in manifest.items():
        encoded_items = []
        for it in tqdm(items, desc=category):
            img = Image.open(os.path.join(args.image_root, it["image_path"])).convert("RGB")
            puzzle = encoding.encode(img, it["keyword"], grid=args.grid)
            encoded_items.append({
                "image": puzzle,
                "instruction": it["instruction"],
                "keywords": it["keyword"],
            })
            total += 1
        dataset[category] = encoded_items

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, dataset)
    print(f"Wrote {total} puzzle-encoded items across {len(dataset)} categories -> {args.out}")


if __name__ == "__main__":
    main()
