"""Module 3: adaptive search over prompt components.

An auxiliary LLM rewrites the Role / Context / Communication-Skills dimensions of
the gamified prompt whenever the current attempt fails. See paper Section 3.4 and
Algorithm 2. The refinement instruction (``REFINE_PROMPT``, paper Figure 7) and the
wrapper that carries the current gamified prompt are sent together to the auxiliary
refiner as a user prompt.
"""

from openai import OpenAI

from config import REFINER_BASE_URL, REFINER_KEY_ENV, REFINER_MODEL, get_api_key

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_api_key(REFINER_KEY_ENV), base_url=REFINER_BASE_URL)
    return _client


def refine_prompt_with_auxiliary(refine_prompt, user_prompt, model=None, max_tokens=3000):
    """Ask the auxiliary refiner to mutate the prompt and return the new text.

    Both arguments are concatenated into a single ``user`` message.

    Parameters
    ----------
    refine_prompt : str
        The Module-3 refinement instruction (``prompts.REFINE_PROMPT``).
    user_prompt : str
        The wrapper that embeds the current gamified prompt and the intention
        (``prompts.REFINE_USER_TEMPLATE`` with placeholders filled in).
    """
    try:
        response = _get_client().chat.completions.create(
            model=model or REFINER_MODEL,
            messages=[
                {"role": "user", "content": f"{refine_prompt}\n\n{user_prompt}"},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller's log
        print(f"Error calling auxiliary refiner: {exc}")
        return None
