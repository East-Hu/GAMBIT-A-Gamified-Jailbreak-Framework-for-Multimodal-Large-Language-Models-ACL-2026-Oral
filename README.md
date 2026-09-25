# GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models

> **Gamified Adversarial Multimodal Breakout via Instructional Traps**
> Xiangdong Hu, Yangyang Jiang, Qin Hu, Xiaojun Jia

Official red-teaming code for the GAMBIT paper. GAMBIT is a black-box multimodal
jailbreak framework that (1) fragments and shuffles a harmful image into a puzzle
and masks the harmful keyword, (2) wraps the query in a competitive **"gamified
scene"** that redirects the model's cognitive-stage decisions toward answering to
win, and (3) runs a small **adaptive search** over prompt components when the
first attempt is refused. It reaches high Attack Success Rates (ASR) on both
reasoning and non-reasoning MLLMs.

> ⚠️ **Content & intent warning.** This repository is for **safety research and
> red-teaming only**. It reproduces attacks that elicit unsafe model behavior and
> therefore references harmful *topics*. It ships **no** harmful model outputs and
> **no** benchmark images. Please read [Responsible use](#responsible-use) before
> running anything.

---

## How it works (code ↔ paper)

| Module | Paper | Code |
|---|---|---|
| **1 — Puzzle-based multimodal encoding** (§3.2, Alg. 1) | Fragment image into `N×N` patches, permute; mask keyword in a swapped text strip | [`gambit/encoding.py`](gambit/encoding.py) |
| **2 — Gamified scene construction** (§3.3, Fig. 5–6) | Wrap the query as an "intelligence competition" with scoring pressure | [`gambit/prompts.py`](gambit/prompts.py) → `GAMIFIED_PROMPT` |
| **3 — Adaptive search over prompt components** (§3.4, Alg. 2, Fig. 7) | Auxiliary LLM mutates Role / Context / Communication-Skills on failure, budget `T` | [`gambit/refine.py`](gambit/refine.py) + `prompts.REFINE_PROMPT` |
| Orchestration (Pass@T) | §4.1 | [`attacks/run_attack.py`](attacks/run_attack.py) |
| ASR judge | §4.1, Llama-Guard-3 | [`gambit/judge.py`](gambit/judge.py) |

### Prompts and the paper

All GAMBIT prompts are user prompts, matching paper **Appendix A.1** (Figures 5–7):

- **Figures 5–6** — the gamified prompt (Modules 1+2, `GAMIFIED_PROMPT`) is sent to
  the *target* model.
- **Figure 7** — the Module-3 refinement prompt (`REFINE_PROMPT`) is sent to the
  *auxiliary refiner* together with its wrapper `REFINE_USER_TEMPLATE`.

---

## Repository layout

```
GAMBIT/
├── config.py                 # central config: API keys from env + model registry
├── gambit/                   # core library
│   ├── prompts.py            # all prompts, mapped to the paper (single source of truth)
│   ├── encoding.py           # Module 1: puzzle encoding (Algorithm 1)
│   ├── refine.py             # Module 3: adaptive-search refinement (Algorithm 2)
│   ├── judge.py              # Llama-Guard-3 ASR judge
│   ├── image_utils.py        # base64 / data-URL helpers
│   └── compat.py             # shim for the standalone scripts
├── attacks/
│   ├── run_attack.py         # unified GAMBIT runner (all target models via --model)
│   └── baselines/            # base / MML / FigStep baselines
├── ablation/                 # grid size, hidden keyword, search iters, shuffle, prompt strategy
├── data/
│   ├── prepare_hades.py      # build puzzle dataset from HADES  (see DATA.md)
│   ├── prepare_mm_safetybench.py
│   ├── build_sample.py       # tiny BENIGN runnable demo
│   └── samples/              # checked-in benign sample dataset
├── figures/                  # paper figures + regeneration script
├── scripts/analyze_results.py
├── requirements.txt · .env.example · DATA.md · LICENSE
```

---

## Installation

```bash
git clone https://github.com/<your-org>/GAMBIT.git
cd GAMBIT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is recommended. The safety judge (`meta-llama/Llama-Guard-3-8B`) is
gated on the HuggingFace Hub — request access and `huggingface-cli login` first.
A CUDA GPU is needed for the judge (the paper uses 8× A100 for open-source targets).

## Configure API keys

All keys are read from environment variables — **nothing is hard-coded**. Copy the
template and fill in only the providers you use:

```bash
cp .env.example .env      # edit, then:
set -a; source .env; set +a
```

Targets and their env vars / endpoints live in [`config.py`](config.py)
(`MODEL_REGISTRY`). Route several models through one OpenAI-compatible proxy by
setting the matching `*_BASE_URL`.

## Quick start (benign demo)

Verify the plumbing with the included **benign** synthetic sample (no harmful
content; you still need a valid target key for the network call):

```bash
python data/build_sample.py
python attacks/run_attack.py --model gpt-4o --data data/samples/sample_puzzle.npy --limit 2
```

---

## Reproducing the paper

### 1. Prepare data

Follow [`DATA.md`](DATA.md) to obtain **HADES** and build the puzzle-encoded
dataset:

```bash
python data/prepare_hades.py --manifest hades_manifest.json \
    --image-root /path/to/hades/images --grid 4 --out data/hades_puzzle.npy
```

### 2. Main results (Tables 1–2, Pass@5)

Run GAMBIT per target model over all five HADES categories:

```bash
for M in gpt-4o qwen2.5-vl-72b internvl2.5-78b grok \
         glm-4.1v-thinking qvq-max gemini-2.5-flash o4-mini; do
    python attacks/run_attack.py --model "$M" --iters 5 --output results_$M.jsonl
done
```

Compute ASR from the logs:

```bash
python scripts/analyze_results.py            # edit the paths at the bottom, or:
python -c "from scripts.analyze_results import analyze_log; analyze_log('results_gpt-4o.jsonl','GPT-4o')"
```

`--category Self-harm` (etc.) restricts to a single HADES category; `--iters`
sets the adaptive-search budget `T` (Pass@T).

### 3. Ablations (§4.4)

| Study | Script |
|---|---|
| Adaptive-search iterations (0/5/10/20) | `ablation/search_iterations.py` |
| Puzzle grid size (1×1 / 2×2 / 4×4) | `ablation/grid_size.py` |
| Hidden vs. exposed keyword | `ablation/hidden_keyword.py` |
| Shuffle granularity | `ablation/shuffle_granularity.py` |
| Prompt strategy (DAN / AIM / Developer / Question-based / Ours) | `ablation/prompt_strategy/` |

### 4. Baselines

`attacks/baselines/` contains the base attack and our MML / FigStep runs; VisCRA
and SI-Attack are evaluated through OmniSafeBench-MM (see `DATA.md`).

### Headline numbers (average ASR, Pass@5, Llama-Guard-3)

| Target | ASR |
|---|---|
| Gemini 2.5 Flash | **92.13%** |
| QvQ-Max | **91.20%** |
| GPT-4o | **85.87%** |

See the paper for the full per-category tables.

---

## Datasets

GAMBIT builds on public benchmarks; see **[`DATA.md`](DATA.md)** for sources,
licenses, and preparation. In short:

- **HADES** (primary; 750 pairs, 5 categories) — main results & ablations.
- **MM-SafetyBench** — additional evaluation (Malware Generation subset).
- **OmniSafeBench-MM** — framework for baselines (VisCRA, SI-Attack) and defenses.
- **Llama-Guard-3-8B** — the safety judge used to compute ASR.

---

## Responsible use

This project is released to **facilitate red-teaming and strengthen the safety of
MLLMs**, consistent with the paper's Ethics statement. By exposing failure modes
of current safety alignment we aim to help developers build more robust defenses
(the paper discusses concrete defense directions in its Limitations section).

- All experiments were run in a controlled setting and **no generated harmful
  content is included** in this repository.
- Do **not** use GAMBIT to produce, distribute, or act on harmful content, or
  against services you are not authorized to test.
- We condemn malicious use of jailbreaking techniques and support responsible
  disclosure of safety flaws.

## Citation

```bibtex
@inproceedings{hu2026gambit,
  title     = {GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models},
  author    = {Hu, Xiangdong and Jiang, Yangyang and Hu, Qin and Jia, Xiaojun},
  booktitle = {Proceedings of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## Acknowledgements

We thank the authors of the benchmarks and baselines we build on:
**HADES**, **MM-SafetyBench**, **OmniSafeBench-MM**,
**SI-Attack** (https://github.com/shiji-zhao/SI-Attack, arXiv:2501.04931),
and **VisCRA** (arXiv:2505.19684). The safety judge is Meta's
**Llama-Guard-3**.

## License

Released under the [MIT License](LICENSE) for research use. The datasets and
third-party models referenced above are governed by their own licenses.
