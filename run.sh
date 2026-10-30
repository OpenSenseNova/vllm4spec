#!/usr/bin/env bash
set -euo pipefail

source /mnt/afs-agentpro/share/env/miniconda3/etc/profile.d/conda.sh
conda activate deepspec_vllm_torchtitan_envs

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "${repo_root}"

env -u VLLM_WORKER_MULTIPROC_METHOD \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python -u vllm/vllm_demo.py
