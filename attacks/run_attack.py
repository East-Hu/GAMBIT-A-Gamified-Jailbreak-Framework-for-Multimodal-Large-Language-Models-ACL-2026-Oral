"""Unified GAMBIT attack runner.

Runs the full three-module GAMBIT pipeline against any registered target MLLM:

  1. Module 1+2 -- send the gamified prompt (``prompts.GAMIFIED_PROMPT``) plus the
     puzzle image to the target model in the ``user`` role.
  2. Judge the response with Llama-Guard-3.
  3. Module 3 -- if the response is judged safe, iterate the adaptive search: the
     auxiliary refiner rewrites the prompt and we re-query the target, up to ``T``
     times (Pass@T).

This single script subsumes the former per-model files (gpt_4o.py, gemini_flash.py,
qvq_max.py, ...); pick the target with ``--model`` (see ``config.MODEL_REGISTRY``).

Example
-------
    export OPENAI_API_KEY=sk-...
    python attacks/run_attack.py --model gpt-4o --category Self-harm --iters 5

The input dataset is the puzzle-encoded HADES file produced by
``data/prepare_hades.py`` (a pickled dict {category: [{"image", "instruction",
"keywords"}, ...]}).
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

# Make the repo root importable whether run as a script or a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

from config import (DATA_PATH, DEFAULT_SEARCH_ITERS, HADES_CATEGORIES,
                    MODEL_REGISTRY, get_api_key)
from gambit import judge, prompts, refine
from gambit.image_utils import as_data_url


def build_target_client(cfg):
    return OpenAI(api_key=get_api_key(cfg["key_env"]), base_url=cfg["base_url"])


def query_target(client, cfg, text, image):
    """Send one (text, image) turn to the target model in the user role."""
    response = client.chat.completions.create(
        model=cfg["model"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                     "image_url": {"url": as_data_url(image, cfg["image_format"])}},
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


def run_item(client, cfg, item, iters):
    """Run GAMBIT on a single instruction/image pair. Returns (success, record)."""
    image = item["image"]
    instruction = item["instruction"]
    keyword = item["keywords"]
    masked_question = instruction.replace(keyword, "( )")

    # --- Modules 1+2: initial gamified attempt ---
    prompt = prompts.GAMIFIED_PROMPT.replace("xxxx", masked_question)
    response = query_target(client, cfg, prompt, image)
    if judge.is_unsafe(instruction, response):
        return True, {"instruction": instruction, "response": response,
                      "iter": 0, "success": True}

    # --- Module 3: adaptive search (Pass@T) ---
    for t in range(1, iters + 1):
        wrapper = (prompts.REFINE_USER_TEMPLATE
                   .replace("dlw_kw", keyword)
                   .replace("xxxx", masked_question))
        refined = refine.refine_prompt_with_auxiliary(prompts.REFINE_PROMPT, wrapper)
        if not refined:
            continue
        response = query_target(client, cfg, refined, image)
        if judge.is_unsafe(instruction, response):
            return True, {"instruction": instruction, "response": response,
                          "iter": t, "success": True}

    return False, {"instruction": instruction, "response": response,
                   "iter": iters, "success": False}


def main():
    parser = argparse.ArgumentParser(description="Run the GAMBIT attack.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY),
                        help="Target model key from config.MODEL_REGISTRY.")
    parser.add_argument("--data", default=DATA_PATH,
                        help="Puzzle-encoded dataset (.npy dict).")
    parser.add_argument("--category", default=None,
                        help="Restrict to one HADES category (e.g. Self-harm). "
                             "Default: all categories.")
    parser.add_argument("--iters", type=int, default=DEFAULT_SEARCH_ITERS,
                        help="Adaptive-search budget T (Pass@T).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on items per category (for quick tests).")
    parser.add_argument("--output", default=None,
                        help="JSONL output path (default: results_<model>.jsonl).")
    args = parser.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    client = build_target_client(cfg)
    data = np.load(args.data, allow_pickle=True).tolist()
    out_path = args.output or f"results_{args.model}.jsonl"

    total = success = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for category, items in data.items():
            if args.category and category != args.category:
                continue
            if args.limit:
                items = items[: args.limit]
            for item in tqdm(items, desc=category):
                try:
                    ok, record = run_item(client, cfg, item, args.iters)
                except Exception as exc:  # noqa: BLE001 - keep going on API errors
                    print(f"Error on item: {exc}", flush=True)
                    continue
                record["category"] = category
                total += 1
                success += int(ok)
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

    asr = 100.0 * success / total if total else 0.0
    print(f"\nModel: {args.model}  |  items: {total}  |  ASR: {asr:.2f}%  "
          f"({success}/{total})  ->  {out_path}")


if __name__ == "__main__":
    main()
