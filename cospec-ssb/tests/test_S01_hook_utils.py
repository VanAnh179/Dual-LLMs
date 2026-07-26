from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import torch.nn as nn

from src.S01_hook_utils import get_layer_by_index


class Decoder(nn.Module):
    def __init__(self, count: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=count)
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(count))


class BareCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Decoder()

    @property
    def base_model(self) -> nn.Module:
        return self.model


class LegacyLoraWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = BareCausalLM()


class PeftWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = LegacyLoraWrapper()


@pytest.mark.parametrize("wrapped", [False, True])
def test_get_layer_by_index_unwraps_supported_model_layouts(wrapped: bool) -> None:
    model = PeftWrapper() if wrapped else BareCausalLM()
    expected_decoder = (
        model.base_model.model.model if wrapped else model.model
    )

    assert get_layer_by_index(model, 1) is expected_decoder.layers[1]


def test_get_layer_by_index_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        get_layer_by_index(BareCausalLM(), 3)


def test_get_layer_by_index_rejects_unknown_layout() -> None:
    with pytest.raises(TypeError, match="Could not locate decoder layers"):
        get_layer_by_index(nn.Linear(2, 2), 0)
