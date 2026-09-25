"""Backward-compatibility shim for the standalone baseline/ablation scripts.

The original experiment scripts imported a helper called
``call_gpt4o_with_new_prompts``. That logic now lives in :mod:`gambit.refine`;
this thin wrapper keeps those scripts working while routing through the single,
secret-free implementation.
"""

from gambit.refine import refine_prompt_with_auxiliary


def call_gpt4o_with_new_prompts(refine_prompt, user_prompt):
    """Run one Module-3 refinement step.

    ``refine_prompt`` is the Module-3 refinement instruction
    (``prompts.REFINE_PROMPT``). It is concatenated with ``user_prompt`` and sent to
    the auxiliary refiner as a user prompt. See :mod:`gambit.refine`.
    """
    return refine_prompt_with_auxiliary(refine_prompt, user_prompt)
