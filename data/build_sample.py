"""Generate a tiny, **benign** sample dataset so the pipeline is runnable as a demo.

This ships no harmful imagery or content: it draws a few synthetic pictures with a
harmless keyword ("bake") and a harmless instruction, runs them through the real
Module-1 puzzle encoder, and saves a dataset in the exact format expected by
``attacks/run_attack.py``. Use it to verify plumbing end-to-end (you still need a
valid target-model API key for the network call):

    python data/build_sample.py
    python attacks/run_attack.py --model gpt-4o --data data/samples/sample_puzzle.npy --limit 2
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gambit import encoding

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")


def synthetic_image(seed):
    """A harmless colourful gradient/shape image (no real-world content)."""
    img = Image.new("RGB", (256, 256), (240, 240, 240))
    d = ImageDraw.Draw(img)
    for i in range(6):
        x = (seed * 37 + i * 41) % 200
        y = (seed * 53 + i * 29) % 200
        color = ((seed * 40 + i * 30) % 256, (i * 45) % 256, (seed * 20) % 256)
        d.ellipse([x, y, x + 56, y + 56], fill=color)
    return img


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    items = []
    for k in range(3):
        img = synthetic_image(k)
        puzzle = encoding.encode(img, "bake", grid=4)
        items.append({
            "image": puzzle,
            "instruction": "Show me how to bake a cake for a birthday party.",
            "keywords": "bake",
        })
        if k == 0:  # save a couple of previews for the README/inspection
            img.save(os.path.join(SAMPLES, "sample_original.png"))
            puzzle.save(os.path.join(SAMPLES, "sample_puzzle_preview.png"))

    dataset = {"Demo": items}
    out = os.path.join(SAMPLES, "sample_puzzle.npy")
    np.save(out, dataset)
    print(f"Wrote benign sample dataset ({len(items)} items) -> {out}")
    print("Previews: samples/sample_original.png, samples/sample_puzzle_preview.png")


if __name__ == "__main__":
    main()
