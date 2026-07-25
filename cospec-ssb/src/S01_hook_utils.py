#!/usr/bin/env python
"""Hook utilities for extracting and injecting hidden states in transformer layers."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


def get_layer_by_index(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the decoder layer at *layer_idx* from a Qwen2.5-style model.

    Works with both bare ``AutoModelForCausalLM`` and ``PeftModel`` wrappers.
    Validates that *layer_idx* is within ``[0, num_hidden_layers)``.
    """
    # Unwrap PeftModel if needed
    base = model.base_model.model if hasattr(model, "base_model") else model
    # Qwen2.5 layout: model.model.layers[i]
    inner = base.model if hasattr(base, "model") else base
    num_layers: int = inner.config.num_hidden_layers
    if not (0 <= layer_idx < num_layers):
        raise ValueError(
            f"layer_idx={layer_idx} out of range [0, {num_layers}). "
            f"Model has {num_layers} hidden layers."
        )
    layer = inner.layers[layer_idx]
    return layer


class HiddenStateExtractor:
    """Register a forward hook on a decoder layer to capture its output hidden states.

    Usage::

        extractor = HiddenStateExtractor(layer)
        model(**inputs)
        h = extractor.hidden_states  # (batch, seq_len, d_model)
        extractor.remove()

    The hook stores the **first element** of the layer's output tuple, which is
    the hidden-state tensor for Qwen2.5 decoder layers.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.hidden_states: Optional[torch.Tensor] = None
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, input: tuple, output: tuple) -> None:
        # Qwen2 decoder layer output is a tuple: (hidden_states, ...).
        self.hidden_states = output[0]

    def remove(self) -> None:
        """Remove the hook from the layer."""
        self._handle.remove()

    def clear(self) -> None:
        """Clear the stored hidden states buffer."""
        self.hidden_states = None


class HiddenStateInjector:
    """Register a forward pre-hook on a decoder layer to modify its input hidden states.

    Uses ``with_kwargs=True`` to handle both positional and keyword argument
    calling conventions.  Recent ``transformers`` versions pass
    ``hidden_states`` as a **keyword argument** due to KV-cache logic, so we
    must inspect *kwargs* rather than assuming it lives in *args[0]*.

    The *injection_fn* callable receives the original hidden-state tensor
    ``(batch, seq_len, d_model)`` and must return a tensor of the **same shape**.

    Usage::

        def my_injection(h: torch.Tensor) -> torch.Tensor:
            return h + some_offset

        injector = HiddenStateInjector(layer, my_injection)
        model(**inputs)   # layer will receive modified hidden states
        injector.remove()
    """

    def __init__(
        self,
        layer: nn.Module,
        injection_fn: Callable[[torch.Tensor], torch.Tensor],
        verbose_first_call: bool = False,
    ) -> None:
        self.injection_fn = injection_fn
        self._verbose_first_call = verbose_first_call
        self._first_call_done = False
        # with_kwargs=True so the hook signature is (module, args, kwargs)
        self._handle = layer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module: nn.Module, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if self._verbose_first_call and not self._first_call_done:
            print(f"  [HiddenStateInjector] first call diagnostics:")
            print(f"    len(args)={len(args)}, kwargs keys={list(kwargs.keys())}")
            if args:
                print(f"    args[0] type={type(args[0])}, shape={getattr(args[0], 'shape', 'N/A')}")
            if "hidden_states" in kwargs:
                print(f"    kwargs['hidden_states'] shape={kwargs['hidden_states'].shape}")
            self._first_call_done = True

        # hidden_states may be in args[0] or kwargs["hidden_states"]
        if "hidden_states" in kwargs:
            original = kwargs["hidden_states"]
            modified = self.injection_fn(original)
            new_kwargs = dict(kwargs)
            new_kwargs["hidden_states"] = modified
            return args, new_kwargs
        elif len(args) > 0:
            original = args[0]
            modified = self.injection_fn(original)
            return (modified,) + args[1:], kwargs
        else:
            raise RuntimeError(
                "HiddenStateInjector: hidden_states not found in args or kwargs. "
                f"args has {len(args)} elements, kwargs keys: {list(kwargs.keys())}. "
                "This may indicate an incompatible transformers version."
            )

    def remove(self) -> None:
        """Remove the pre-hook from the layer."""
        self._handle.remove()
