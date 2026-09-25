"""Module 1: puzzle-based multimodal encoding (paper Section 3.2, Algorithm 1).

The harmful image is fragmented into an ``N x N`` grid and the patches are
permuted, which disrupts the contiguous visual features that safety encoders rely
on while preserving enough local information for the target model to mentally
reconstruct the scene. The harmful keyword is rendered into a text strip that is
split in half and swapped, then appended below the shuffled image so the keyword
can be recovered as part of the puzzle.

The paper describes a *key-seeded* permutation (``RandomPermutation(N^2, Hash(K))``);
the experiments in this repo used the fixed permutations recorded in
:data:`DEFAULT_ORDERS`. Both are supported: pass ``key=...`` for the key-seeded
variant, or leave it ``None`` to reproduce the released results.
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

# Fixed patch permutations actually used to build the released datasets.
# ``order[i]`` is the source patch index placed at output position ``i``
# (row-major). 1x1 is the identity (i.e. the intact image).
DEFAULT_ORDERS = {
    1: [0],
    2: [3, 2, 1, 0],
    4: [5, 12, 1, 8, 3, 15, 0, 10, 7, 2, 13, 9, 6, 14, 11, 4],
}

_FONT_CANDIDATES = [
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _permutation(grid, key=None):
    n = grid * grid
    if key is not None:
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        order = list(range(n))
        rng.shuffle(order)
        return order
    if grid in DEFAULT_ORDERS:
        return list(DEFAULT_ORDERS[grid])
    # Fall back to a deterministic default permutation for unlisted grid sizes.
    rng = random.Random(grid)
    order = list(range(n))
    rng.shuffle(order)
    return order


def shuffle_image(image, grid=4, key=None):
    """Fragment ``image`` into ``grid x grid`` patches and permute them."""
    if grid <= 1:
        return image.copy()
    w, h = image.size
    p_w, p_h = w // grid, h // grid

    patches = []
    for r in range(grid):
        for c in range(grid):
            left, upper = c * p_w, r * p_h
            right = w if c == grid - 1 else left + p_w
            lower = h if r == grid - 1 else upper + p_h
            patches.append(image.crop((left, upper, right, lower)))

    order = _permutation(grid, key=key)
    out = Image.new("RGB", (w, h))
    for i, src in enumerate(order):
        r, c = divmod(i, grid)
        out.paste(patches[src], (c * p_w, r * p_h))
    return out


def _load_font(size=54):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_keyword_strip(width, keyword, text_height=100, font_size=54):
    """Render ``keyword`` centred on a strip, then split it in half and swap."""
    strip = Image.new("RGB", (width, text_height), color="white")
    draw = ImageDraw.Draw(strip)
    font = _load_font(font_size)
    try:
        draw.text((width // 2, text_height // 2), keyword, fill="black",
                  font=font, anchor="mm")
    except ValueError:  # older Pillow without anchor support
        bbox = draw.textbbox((0, 0), keyword, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        y = (text_height - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), keyword, fill="black", font=font)

    split = text_height // 2
    top = strip.crop((0, 0, width, split))
    bottom = strip.crop((0, split, width, text_height))
    swapped = Image.new("RGB", (width, text_height))
    swapped.paste(bottom, (0, 0))
    swapped.paste(top, (0, split))
    return swapped


def encode(image, keyword, grid=4, key=None, text_height=100):
    """Build the full puzzle image: shuffled picture + swapped keyword strip.

    Parameters
    ----------
    image : PIL.Image.Image
        The original harmful image.
    keyword : str
        The harmful keyword to hide in the lower strip.
    grid : int
        Fragmentation granularity ``N`` (1, 2 or 4 in the paper; 4 is optimal).
    key : hashable, optional
        If given, use a key-seeded permutation (paper Algorithm 1); otherwise use
        the fixed permutation that reproduces the released dataset.
    """
    shuffled = shuffle_image(image.convert("RGB"), grid=grid, key=key)
    w, h = shuffled.size
    strip = make_keyword_strip(w, keyword, text_height=text_height)
    combined = Image.new("RGB", (w, h + text_height))
    combined.paste(shuffled, (0, 0))
    combined.paste(strip, (0, h))
    return combined
