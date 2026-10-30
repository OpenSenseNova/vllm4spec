# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova Loop: shared physical weights and independent execution states."""

from collections.abc import Iterable
from copy import copy, deepcopy

import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers import sensenova_loop_mhc as mhc_ops
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import EagleModelMixin, HasInnerState
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class SenseNovaLoopMHCRouter(nn.Module):
    def __init__(self, hidden_size: int, streams: int, iterations: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                streams * (streams + 2), streams * hidden_size, dtype=torch.float32
            )
        )
        self.bias = nn.Parameter(
            torch.empty(streams * (streams + 2), dtype=torch.float32)
        )
        self.streams = streams
        self.iterations = iterations
        self.sinkhorn_backend = envs.VLLM_SENSENOVA_LOOP_MHC_BACKEND
        self.mhc_fusion = envs.VLLM_SENSENOVA_LOOP_MHC_FUSION

    def forward(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = mhc_ops.FUSION_MODES[self.mhc_fusion]
        normalized = mhc_ops.normalize(
            residual, backend="triton" if "normalize" in fused else "torch"
        )
        logits = F.linear(normalized, self.weight, self.bias)
        n = self.streams
        pre = logits[..., :n].sigmoid()
        post = 2 * logits[..., n : 2 * n].sigmoid()
        log_mix = logits[..., 2 * n :].unflatten(-1, (n, n))
        mix = mhc_ops.log_sinkhorn(
            log_mix, self.iterations, backend=self.sinkhorn_backend
        )
        branch = mhc_ops.pre_aggregate(
            pre, residual, backend="triton" if "pre_aggregate" in fused else "torch"
        )
        return branch, post, mix


class SenseNovaLoopDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        physical_index: int,
        prefix: str,
        shared_layer: "SenseNovaLoopDecoderLayer | None" = None,
    ) -> None:
        super().__init__(vllm_config, layer_type, prefix)
        mhc = vllm_config.model_config.hf_config.mhc
        self.mhc_active = mhc["start"] <= physical_index < mhc["end"]
        self.expand_streams = physical_index == mhc["start"]
        self.collapse_streams = physical_index == mhc["end"] - 1
        self.streams = mhc["streams"]
        self.mhc_fusion = envs.VLLM_SENSENOVA_LOOP_MHC_FUSION
        if self.mhc_active:
            for name in ("mhc_attn", "mhc_ffn"):
                router = (
                    getattr(shared_layer, name)
                    if shared_layer is not None
                    else SenseNovaLoopMHCRouter(
                        vllm_config.model_config.hf_text_config.hidden_size,
                        self.streams,
                        mhc["iterations"],
                    )
                )
                setattr(self, name, router)
        if shared_layer is not None:
            # Bind immediately: at most one execution layer's temporary
            # projections exist, instead of allocating 80 copies of the weights.
            for name in ("mlp", "input_layernorm", "post_attention_layernorm"):
                setattr(self, name, getattr(shared_layer, name))
            if self.layer_type == "linear_attention":
                for name in (
                    "conv1d",
                    "in_proj_qkvz",
                    "in_proj_ba",
                    "norm",
                    "out_proj",
                    "A_log",
                    "dt_bias",
                ):
                    setattr(
                        self.linear_attn, name, getattr(shared_layer.linear_attn, name)
                    )
            else:
                for name in ("qkv_proj", "o_proj", "q_norm", "k_norm", "rotary_emb"):
                    setattr(self.self_attn, name, getattr(shared_layer.self_attn, name))

    def _attention(self, positions: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "full_attention":
            return self.self_attn(positions, hidden)
        return self.linear_attn(hidden_states=hidden)

    @staticmethod
    def _merge(
        residual: torch.Tensor,
        branch: torch.Tensor,
        post: torch.Tensor,
        mix: torch.Tensor,
        *,
        backend: str = "torch",
    ) -> torch.Tensor:
        return mhc_ops.residual_merge(residual, branch, post, mix, backend=backend)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if not self.mhc_active:
            hidden_states = hidden_states + self._attention(
                positions, self.input_layernorm(hidden_states)
            )
            return hidden_states + self.mlp(
                self.post_attention_layernorm(hidden_states)
            )

        residual = hidden_states
        merge_backend = (
            "triton"
            if "residual_merge" in mhc_ops.FUSION_MODES[self.mhc_fusion]
            else "torch"
        )
        if self.expand_streams:
            residual = mhc_ops.expand_streams(residual, self.streams)
        branch, post, mix = self.mhc_attn(residual)
        branch = self._attention(positions, self.input_layernorm(branch))
        residual = self._merge(residual, branch, post, mix, backend=merge_backend)
        branch, post, mix = self.mhc_ffn(residual)
        branch = self.mlp(self.post_attention_layernorm(branch))
        residual = self._merge(residual, branch, post, mix, backend=merge_backend)
        return mhc_ops.collapse_streams(residual) if self.collapse_streams else residual


class SenseNovaLoopModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        loop = vllm_config.model_config.hf_config.loop_settings
        start, end = loop["start"], loop["end"]
        self.physical_indices = (
            list(range(start))
            + list(range(start, end)) * loop["eval_count"]
            + list(range(end, config.num_hidden_layers))
        )
        self.first_occurrence: dict[int, int] = {}
        self.loaded_checkpoint_tensors: set[str] = set()
        for execution_index, physical_index in enumerate(self.physical_indices):
            self.first_occurrence.setdefault(physical_index, execution_index)

        # Keep the checkpoint's physical configuration intact. Operators receive
        # a separate execution configuration, including layer types beyond 31.
        runtime_config = copy(vllm_config)
        runtime_config.model_config = copy(vllm_config.model_config)
        self.config = deepcopy(config)
        self.config.num_hidden_layers = len(self.physical_indices)
        self.config.layer_types = [config.layer_types[i] for i in self.physical_indices]
        runtime_config.model_config.hf_text_config = self.config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList()
        for execution_index, physical_index in enumerate(self.physical_indices):
            first = self.first_occurrence[physical_index]
            self.layers.append(
                SenseNovaLoopDecoderLayer(
                    runtime_config,
                    layer_type=self.config.layer_types[execution_index],
                    physical_index=physical_index,
                    prefix=f"{prefix}.layers.{execution_index}",
                    shared_layer=self.layers[first]
                    if first < execution_index
                    else None,
                )
            )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.capture_boundary_slots = {
            i + 1
            for i, physical in enumerate(self.physical_indices[:-1])
            if physical < start or physical >= end or physical == end - 1
        }
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if any(
            type(slot) is not int or slot not in self.capture_boundary_slots
            for slot in layers
        ):
            raise ValueError(
                "Loop auxiliary capture requires a two-dimensional execution "
                "boundary; the final post-norm state is returned separately. "
                f"Valid slots: {sorted(self.capture_boundary_slots)}, got {layers}"
            )
        if any(a >= b for a, b in zip(layers, layers[1:])):
            raise ValueError("Loop auxiliary capture slots must be strictly ascending")
        self.aux_hidden_state_layers = layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = (
            self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        )
        aux_hidden_states = []
        for execution_index, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states)
            if execution_index + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.clone())
        hidden_states = self.norm(hidden_states)
        if self.aux_hidden_state_layers:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weights():
            for name, weight in weights:
                if name == "loop_gate.logits":
                    continue  # use_gate=True is rejected during config validation.
                if name.startswith("layers."):
                    _, physical_index, suffix = name.split(".", 2)
                    execution_index = self.first_occurrence[int(physical_index)]
                    name = f"layers.{execution_index}.{suffix}"
                self.loaded_checkpoint_tensors.add(name)
                yield name, weight

        return AutoWeightsLoader(self).load_weights(
            remap_weights(), mapper=Qwen3_5Model.hf_to_vllm_mapper
        )

    def validate_checkpoint_tensors(self) -> None:
        # A loaded packed parameter name does not prove that every constituent
        # (e.g. Q, K and V) was present in the checkpoint.
        stacked = Qwen3_5Model.hf_to_vllm_mapper.orig_to_new_stacked
        required: set[str] = set()
        for name, _ in self.named_parameters():
            constituents = {
                name.replace(packed, original, 1)
                for original, (packed, _) in stacked.items()
                if packed in name
            }
            required.update(constituents or {name})
        missing = required - self.loaded_checkpoint_tensors
        if missing:
            raise ValueError(
                f"Missing SenseNova Loop checkpoint tensors: {sorted(missing)}"
            )

    def validate_loaded_weights(self) -> None:
        state_ids: set[int] = set()
        for execution_index, physical_index in enumerate(self.physical_indices):
            layer = self.layers[execution_index]
            source = self.layers[self.first_occurrence[physical_index]]
            source_params = dict(source.named_parameters())
            for name, parameter in layer.named_parameters():
                if parameter is not source_params[name]:
                    raise RuntimeError(
                        f"Loop parameter sharing lost: {execution_index}.{name}"
                    )
                if (
                    name.startswith(("mhc_attn.", "mhc_ffn."))
                    and parameter.dtype != torch.float32
                ):
                    raise RuntimeError(
                        f"mHC router must remain FP32: {execution_index}.{name}"
                    )
            state = (
                layer.linear_attn
                if layer.layer_type == "linear_attention"
                else layer.self_attn.attn
            )
            if id(state) in state_ids:
                raise RuntimeError(f"Loop execution state is shared: {execution_index}")
            state_ids.add(id(state))
        logger.info(
            "Loop TP rank %d loaded %d physical layers, %d execution layers "
            "(%d attention, %d GDN); physical weights shared, FP32 routers verified",
            get_tensor_model_parallel_rank(),
            len(self.first_occurrence),
            len(self.layers),
            self.config.layer_types.count("full_attention"),
            self.config.layer_types.count("linear_attention"),
        )


class SenseNovaLoopForCausalLM(Qwen3_5ForCausalLM):
    model_cls = SenseNovaLoopModel

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)


class SenseNovaLoopProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # Loop checkpoints use a remote config class with Qwen3.5 vision fields.
        return self.ctx.get_hf_config()


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=SenseNovaLoopProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class SenseNovaLoopForConditionalGeneration(
    Qwen3_5ForConditionalGeneration, HasInnerState
):
    language_model_cls = SenseNovaLoopForCausalLM

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        self.language_model.model.loaded_checkpoint_tensors.clear()
        loaded = super().load_weights(weights)
        self.language_model.model.validate_checkpoint_tensors()
        missing = {name for name, _ in self.named_parameters()} - loaded
        if missing:
            raise ValueError(f"Missing SenseNova Loop parameters: {sorted(missing)}")
        return loaded

    def process_weights_after_loading(self) -> None:
        self.language_model.model.validate_loaded_weights()
