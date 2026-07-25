#!/usr/bin/env python
"""Sequential A→B runner for memory-efficient evaluation when both models cannot fit in VRAM simultaneously."""
from __future__ import annotations

import gc
from typing import Any, Callable, Dict, Optional

import torch


def clear_model(*objects: Any) -> None:
    """Delete model objects and free GPU memory."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential(
    load_a_fn: Callable[[], Any],
    forward_a_fn: Callable[[Any], Dict[str, Any]],
    load_b_fn: Callable[[], Any],
    forward_b_fn: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    clear_between: bool = True,
) -> Dict[str, Any]:
    """Run Agent A then Agent B sequentially, optionally clearing VRAM between them.

    Parameters
    ----------
    load_a_fn : callable
        Returns loaded Agent A resources (model, tokenizer, etc.) as any object.
    forward_a_fn : callable
        Takes Agent A resources, runs forward, returns a dict of results.
        If *clear_between* is True, any tensors that need to survive must be
        ``.detach().cpu()``'d inside this function.
    load_b_fn : callable
        Returns loaded Agent B resources.
    forward_b_fn : callable
        Takes Agent B resources and the dict returned by *forward_a_fn*,
        runs forward, returns a dict of final results.
    clear_between : bool
        If True, delete Agent A resources and clear VRAM before loading B.

    Returns
    -------
    dict
        Combined results: ``{"a_results": ..., "b_results": ...}``.
    """
    # --- Agent A ---
    a_resources = load_a_fn()
    a_results = forward_a_fn(a_resources)

    if clear_between:
        clear_model(a_resources)
    else:
        # Keep reference alive but don't explicitly clear
        del a_resources

    # --- Agent B ---
    b_resources = load_b_fn()
    b_results = forward_b_fn(b_resources, a_results)

    clear_model(b_resources)

    return {"a_results": a_results, "b_results": b_results}
