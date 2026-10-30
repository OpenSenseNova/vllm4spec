#!/usr/bin/env bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bench_dir/../../.." && pwd)
cd "$repo_dir"
# shellcheck source=/dev/null
source "$repo_dir/h800conda-tmp.sh"
export TARGET_MODEL_PATH=/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_SENSENOVA_LOOP_MHC_BACKEND=triton
export VLLM_SENSENOVA_LOOP_MHC_FUSION=all
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/loop-autoresearch-20261023/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/loop-autoresearch-20261023/inductor
export SENSESPEC_LOG_DIR="$repo_dir/outputs/loop-autoresearch-20261023/logs"
exec "$PIPELINE_PYTHON" "$bench_dir/verify.py" "$@"
