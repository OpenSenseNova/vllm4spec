# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate a local GLM DSpark cache, optionally running a full draft update.

Requires the adjacent DeepSpec checkout and its training dependencies.
"""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.extend([str(root), str(root.parent)])

    import torch
    import torch.nn.functional as F
    from deepspec.data.cuda_prefetcher import move_batch_to_device
    from deepspec.data.target_cache_dataset import (
        CacheCollator,
        CacheDataset,
        validate_train_cache,
    )
    from deepspec.modeling.dspark.glm5_next import (
        Glm5NextDSparkModel,
        build_draft_config,
    )
    from deepspec.modeling.dspark.loss import compute_dspark_loss
    from deepspec.utils import load_config
    from safetensors import safe_open
    from transformers import AutoConfig

    from vllm_demo import SOURCE_MODEL_PATH, resolve_model_path

    torch.set_num_threads(8)
    cache = CacheDataset(str(args.cache))
    print(f"CacheDataset: {len(cache)} samples", flush=True)
    if args.reference_cache:
        reference = CacheDataset(str(args.reference_cache))
        assert cache.target_layer_ids == reference.target_layer_ids
        assert cache.hidden_size == reference.hidden_size
        for i in range(min(len(cache), len(reference))):
            sample, expected = cache[i], reference[i]
            for key in ("input_ids", "loss_mask"):
                torch.testing.assert_close(sample[key], expected[key])
            for offset, layer in enumerate(cache.target_layer_ids + [44]):
                if offset == len(cache.target_layer_ids):
                    actual = sample["target_last_hidden_states"].float()
                    target = expected["target_last_hidden_states"].float()
                else:
                    span = slice(offset * 4096, (offset + 1) * 4096)
                    actual = sample["target_hidden_states"][:, span].float()
                    target = expected["target_hidden_states"][:, span].float()
                print(
                    json.dumps(
                        {
                            "sample": i,
                            "layer": layer,
                            "cosine_mean": F.cosine_similarity(actual, target)
                            .mean()
                            .item(),
                            "relative_l2": (
                                (actual - target).norm() / target.norm()
                            ).item(),
                            "max_abs": (actual - target).abs().max().item(),
                        }
                    ),
                    flush=True,
                )
        reference.close()

    if args.train_step:
        started = time.monotonic()
        torch.manual_seed(0)
        checkpoint = resolve_model_path()
        config = load_config(root.parent / "config/dspark/dspark_glm5_3_flash.py")
        config.model.target_layer_ids = cache.target_layer_ids
        draft_config = build_draft_config(
            AutoConfig.from_pretrained(checkpoint, local_files_only=True), config.model
        )
        print(
            "Initializing the full 3-layer, 288-expert BF16 DSpark draft.", flush=True
        )
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                model = Glm5NextDSparkModel(draft_config)
        finally:
            torch.set_default_dtype(previous_dtype)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
        for name, parameter in (
            ("model.language_model.embed_tokens.weight", model.embed_tokens.weight),
            ("lm_head.weight", model.lm_head.weight),
        ):
            with (
                safe_open(
                    checkpoint / index["weight_map"][name], framework="pt"
                ) as handle,
                torch.no_grad(),
            ):
                parameter.copy_(handle.get_tensor(name))
        model.set_embedding_head_trainable(False)
        validate_train_cache(
            train_dataset=cache,
            draft_model=model,
            target_model_name_or_path=str(SOURCE_MODEL_PATH),
            chat_template="glm5_next",
            max_length=cache.manifest["max_length"],
            stores_target_last_hidden_states=True,
        )
        batch = CacheCollator()([cache[0]])
        batch = move_batch_to_device(batch, "cuda")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, foreach=False)
        before = model.fc.weight.detach().clone()
        output = model(
            **{
                key: batch[key]
                for key in (
                    "input_ids",
                    "loss_mask",
                    "target_hidden_states",
                    "target_last_hidden_states",
                    "context_start",
                    "context_len",
                    "seq_len",
                )
            }
        )
        loss = compute_dspark_loss(
            outputs=output,
            loss_decay_gamma=4.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            confidence_head_alpha=1.0,
        )
        assert torch.isfinite(loss), "Non-finite training loss"
        loss.backward()
        gradients = [p.grad for p in parameters if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        assert not torch.equal(before, model.fc.weight), "Optimizer did not update fc"
        print(
            json.dumps(
                {
                    "train_step": "passed",
                    "loss": loss.item(),
                    "tokens": batch["input_ids"].shape[1],
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "tensors_with_gradients": len(gradients),
                    "peak_gpu_gib": torch.accelerator.max_memory_allocated() / 1024**3,
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    cache.close()


if __name__ == "__main__":
    main()
