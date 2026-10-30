# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
from transformers import Qwen3_5Config

from vllm import EngineArgs
from vllm.config import ModelConfig


def test_sensenova_vision_accepts_checkpoint_config_class():
    from vllm.model_executor.models.sensenova_loop import SenseNovaLoopProcessingInfo
    from vllm.multimodal.processing.context import InputProcessingContext

    class RemoteLoopConfig(Qwen3_5Config):
        model_type = "sensenova_loop"

    config = RemoteLoopConfig()
    info = SenseNovaLoopProcessingInfo(
        InputProcessingContext(Mock(spec=ModelConfig, hf_config=config), tokenizer=None)
    )
    assert info.get_hf_config() is config


def _engine_args(tmp_path, streams, pre_scale=1.0, use_gate=None):
    loop_settings = {"start": 4, "end": 28, "eval_count": 2}
    if use_gate is not None:
        loop_settings["use_gate"] = use_gate
    config = Qwen3_5Config(
        architectures=["SenseNovaLoopForConditionalGeneration"],
        text_config={"num_hidden_layers": 32},
        loop_settings=loop_settings,
        mhc={
            "start": 4,
            "end": 28,
            "streams": streams,
            "iterations": 20,
            "fixed_identity": False,
            "pre_scale": pre_scale,
        },
    )
    config.save_pretrained(tmp_path)
    return EngineArgs(
        model=str(tmp_path),
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        dtype="bfloat16",
        enforce_eager=True,
        language_model_only=True,
        max_model_len=4096,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        mamba_ssm_cache_dtype="float32",
    )


@pytest.mark.parametrize("streams", [2, 4])
@pytest.mark.parametrize("use_gate", [None, False])
def test_loop_config_preserves_physical_depth(tmp_path, monkeypatch, streams, use_gate):
    """Omitted and disabled gates accept shared weights at the physical depth."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _engine_args(tmp_path, streams, use_gate=use_gate).create_engine_config()
    assert config.model_config.hf_text_config.num_hidden_layers == 32
    assert config.model_config.hf_config.mhc["streams"] == streams


@pytest.mark.parametrize("streams,pre_scale", [(3, 1.0), (2, 0.5)])
def test_sensenova_rejects_unimplemented_routing(
    tmp_path, monkeypatch, streams, pre_scale
):
    """Unsupported routing must fail instead of silently changing the model."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    args = _engine_args(tmp_path, streams, pre_scale)
    with pytest.raises(ValueError, match="two or four.*pre_scale=1.0"):
        args.create_engine_config()


def test_sensenova_rejects_enabled_loop_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    args = _engine_args(tmp_path, 2, use_gate=True)
    with pytest.raises(ValueError, match="without layer_scale or Loop gate"):
        args.create_engine_config()
