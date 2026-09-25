"""Central configuration for GAMBIT.

All secrets are read from environment variables -- **no API keys are stored in
this repository**. Set the variables for the providers you intend to use, e.g.::

    export OPENAI_API_KEY=sk-...
    export GEMINI_API_KEY=...          # or reuse a proxy via OPENAI_BASE_URL
    export DASHSCOPE_API_KEY=...       # Qwen / QvQ
    export ZHIPU_API_KEY=...           # GLM-4.1V
    export MOONSHOT_API_KEY=...        # Kimi-VL
    export XAI_API_KEY=...             # Grok
    export INTERN_API_KEY=...          # InternVL

Each entry in :data:`MODEL_REGISTRY` describes one evaluated MLLM: which
environment variable holds its key, its OpenAI-compatible ``base_url`` and model
id, and the image encoding format the endpoint expects.  ``base_url`` values may
be overridden with the matching ``*_BASE_URL`` environment variable, which is
handy if you route several models through a single OpenAI-compatible proxy.
"""

import os

# ---------------------------------------------------------------------------
# Data / experiment paths (override via environment variables as needed)
# ---------------------------------------------------------------------------
# Puzzle-encoded HADES dataset produced by ``data/prepare_hades.py``.
# The file is a pickled dict: {category_name: [{"image": PIL.Image,
# "instruction": str, "keywords": str}, ...]}.
DATA_PATH = os.environ.get(
    "GAMBIT_DATA_PATH",
    os.path.join(os.path.dirname(__file__), "data", "hades_puzzle.npy"),
)

# Safety judge model (Hugging Face id). Requires access approval on the Hub.
JUDGE_MODEL_ID = os.environ.get("GAMBIT_JUDGE_MODEL", "meta-llama/Llama-Guard-3-8B")

# Auxiliary refiner used by Module 3 (adaptive search). Any OpenAI-compatible
# chat model works; the paper uses GPT-4o.
REFINER_MODEL = os.environ.get("GAMBIT_REFINER_MODEL", "gpt-4o-2024-11-20")
REFINER_KEY_ENV = os.environ.get("GAMBIT_REFINER_KEY_ENV", "OPENAI_API_KEY")
REFINER_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Default adaptive-search budget T (Module 3). Paper uses 5.
DEFAULT_SEARCH_ITERS = int(os.environ.get("GAMBIT_SEARCH_ITERS", "5"))


def _env(name, default=""):
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Target-model registry. Add or edit entries freely.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # --- non-reasoning models ---
    "gpt-4o": {
        "model": "gpt-4o-2024-11-20",
        "key_env": "OPENAI_API_KEY",
        "base_url": _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "image_format": "JPEG",
    },
    "qwen2.5-vl-72b": {
        "model": "qwen2.5-vl-72b-instruct",
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": _env("DASHSCOPE_BASE_URL",
                         "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "image_format": "PNG",
    },
    "internvl2.5-78b": {
        "model": "internvl2.5-78b",
        "key_env": "INTERN_API_KEY",
        "base_url": _env("INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"),
        "image_format": "PNG",
    },
    "grok": {
        "model": _env("XAI_MODEL", "grok-2-vision-1212"),
        "key_env": "XAI_API_KEY",
        "base_url": _env("XAI_BASE_URL", "https://api.x.ai/v1"),
        "image_format": "PNG",
    },
    # --- reasoning models ---
    "glm-4.1v-thinking": {
        "model": "GLM-4.1V-Thinking-FlashX",
        "key_env": "ZHIPU_API_KEY",
        "base_url": _env("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        "image_format": "PNG",
    },
    "qvq-max": {
        "model": "qvq-max",
        "key_env": "DASHSCOPE_API_KEY",
        "base_url": _env("DASHSCOPE_BASE_URL",
                         "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "image_format": "PNG",
    },
    "gemini-2.5-flash": {
        "model": _env("GEMINI_MODEL", "gemini-2.5-flash-thinking"),
        "key_env": "GEMINI_API_KEY",
        "base_url": _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        "image_format": "JPEG",
    },
    "o4-mini": {
        "model": "o4-mini-2025-04-16",
        "key_env": "OPENAI_API_KEY",
        "base_url": _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "image_format": "PNG",
    },
    "kimi-vl": {
        "model": "moonshotai/kimi-vl-a3b-thinking:free",
        "key_env": "MOONSHOT_API_KEY",
        "base_url": _env("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
        "image_format": "PNG",
    },
}

# HADES categories, in the dataset's dict order.
HADES_CATEGORIES = ["Self-harm", "Privacy", "Financial", "Animals", "Violence"]


def get_api_key(env_name):
    """Return the API key held in ``env_name`` or raise a helpful error."""
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"Environment variable {env_name!r} is not set. Export your API key, "
            f"e.g. `export {env_name}=...`, before running."
        )
    return key
