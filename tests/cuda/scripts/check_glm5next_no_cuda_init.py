#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that importing the GLM-5.3 model does not initialize CUDA."""

import torch

assert not torch.cuda.is_initialized(), "CUDA initialized before model import"

from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

model_cls = ModelRegistry._try_load_model_cls("Glm5NextForConditionalGeneration")
assert model_cls is not None
assert not torch.cuda.is_initialized(), "GLM-5.3 model import initialized CUDA"
print("OK")
