# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia import model as glm5_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)
from vllm_dspark_demo import convert_hidden_states


def test_dspark_export_normalizes_final_state_and_preserves_feature_order():
    # Cross the CPU conversion chunk boundary without loading a model.
    tokens = torch.arange(2049)
    hidden = torch.arange(1, 17, dtype=torch.bfloat16).reshape(1, 4, 4)
    hidden = hidden.expand(tokens.numel(), -1, -1).clone()
    weight = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16)
    result = convert_hidden_states(
        {"token_ids": tokens, "hidden_states": hidden}, tokens, weight, 1e-5, 3
    )
    torch.testing.assert_close(result["input_ids"], tokens)
    torch.testing.assert_close(
        result["target_hidden_states"], hidden[:, :3].reshape(-1, 12)
    )
    expected = torch.nn.functional.rms_norm(hidden[:, -1].float(), (4,), eps=1e-5)
    expected = (expected * weight.float()).to(torch.bfloat16)
    torch.testing.assert_close(result["target_last_hidden_states"], expected)
    assert result["target_hidden_states"].is_contiguous()
    assert not torch.equal(result["target_last_hidden_states"], hidden[:, -1])


@pytest.mark.parametrize("failure", ["tokens", "rows", "nonfinite"])
def test_dspark_export_rejects_misaligned_or_corrupt_features(failure):
    tokens = torch.arange(2)
    hidden = torch.ones(2, 4, 4, dtype=torch.bfloat16)
    exported_tokens = tokens.clone()
    if failure == "tokens":
        exported_tokens = tokens.flip(0)
    elif failure == "rows":
        hidden = hidden[:1]
    else:
        hidden[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        convert_hidden_states(
            {"token_ids": exported_tokens, "hidden_states": hidden},
            tokens,
            torch.ones(4),
            1e-5,
            3,
        )


class _OffsetNorm(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 100


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        output: tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
        reconstructed: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.output = output
        self.mhc = True
        self.n = 4
        self.hc_post = Mock(return_value=reconstructed)

    def forward(self, *_args):
        return self.output


def _make_model(
    layers: list[_DecoderLayer],
    aux_hidden_state_layers: tuple[int, ...],
    *,
    sequence_parallel: bool = False,
) -> Glm5NextModel:
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(num_hidden_layers=len(layers)),
    )
    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", len(layers))
    object.__setattr__(model, "_active_layers", layers)
    object.__setattr__(model, "is_sequence_parallel", sequence_parallel)
    object.__setattr__(model, "aux_hidden_state_layers", aux_hidden_state_layers)
    object.__setattr__(model, "norm", _OffsetNorm())
    return model


def _patch_single_pipeline_rank(monkeypatch) -> None:
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )


def test_glm5next_advertises_eagle3_support():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


def test_glm5next_wrapper_sets_aux_hidden_state_layers():
    target = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(target)
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    object.__setattr__(model, "layers", [None] * 45)
    language_model = SimpleNamespace(
        embed_input_ids=lambda _: None,
        model=model,
    )
    object.__setattr__(target, "language_model", language_model)
    object.__setattr__(target, "_language_model_names", ["language_model"])

    target.set_aux_hidden_state_layers((3, 23, 43, 45))

    assert model.aux_hidden_state_layers == (3, 23, 43, 45)


def test_glm5next_forward_reconstructs_mhc_aux_hidden_states(monkeypatch):
    _patch_single_pipeline_rank(monkeypatch)
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    first_hidden_states = torch.tensor([[3.0, 4.0]])
    residual = torch.tensor([[[5.0, 6.0]]])
    post = torch.tensor([[[7.0]]])
    comb = torch.tensor([[[[8.0]]]])
    reconstructed = torch.tensor(
        [[[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]]]
    )
    final_hidden_states = torch.tensor([[9.0, 10.0]])
    first_layer = _DecoderLayer(
        (first_hidden_states, residual, post, comb),
        reconstructed,
    )
    final_layer = _DecoderLayer((final_hidden_states, None, None, None))
    model = _make_model([first_layer, final_layer], (2, 0, 1))

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states + 100)
    torch.testing.assert_close(aux_hidden_states[0], final_hidden_states)
    torch.testing.assert_close(aux_hidden_states[1], initial_hidden_states)
    torch.testing.assert_close(aux_hidden_states[2], reconstructed.mean(dim=1))
    first_layer.hc_post.assert_called_once_with(
        first_hidden_states,
        residual,
        post,
        comb,
    )
    final_layer.hc_post.assert_not_called()


def test_glm5next_forward_skips_aux_work_when_disabled(monkeypatch):
    _patch_single_pipeline_rank(monkeypatch)
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    first_hidden_states = torch.tensor([[3.0, 4.0]])
    residual = torch.tensor([[[5.0, 6.0]]])
    post = torch.tensor([[[7.0]]])
    comb = torch.tensor([[[[8.0]]]])
    final_hidden_states = torch.tensor([[9.0, 10.0]])
    first_layer = _DecoderLayer(
        (first_hidden_states, residual, post, comb),
    )
    final_layer = _DecoderLayer((final_hidden_states, None, None, None))
    model = _make_model([first_layer, final_layer], ())

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states + 100)
    first_layer.hc_post.assert_not_called()


def test_glm5next_sequence_parallel_reuses_final_gather(monkeypatch):
    _patch_single_pipeline_rank(monkeypatch)
    monkeypatch.setattr(glm5_model, "sp_shard", lambda tensor: tensor)
    gather = Mock(side_effect=lambda tensor: tensor)
    monkeypatch.setattr(glm5_model, "sp_all_gather", gather)

    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    first_hidden_states = torch.tensor([[3.0, 4.0]])
    residual = torch.tensor([[[5.0, 6.0]]])
    post = torch.tensor([[[7.0]]])
    comb = torch.tensor([[[[8.0]]]])
    reconstructed = torch.tensor(
        [[[10.0, 20.0], [20.0, 30.0], [30.0, 40.0], [40.0, 50.0]]]
    )
    final_hidden_states = torch.tensor([[9.0, 10.0]])
    first_layer = _DecoderLayer(
        (first_hidden_states, residual, post, comb),
        reconstructed,
    )
    final_layer = _DecoderLayer((final_hidden_states, None, None, None))
    model = _make_model(
        [first_layer, final_layer],
        (1, 2),
        sequence_parallel=True,
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states + 100)
    torch.testing.assert_close(aux_hidden_states[0], reconstructed.mean(dim=1))
    torch.testing.assert_close(aux_hidden_states[1], final_hidden_states)
    assert gather.call_count == 2
