#!/usr/bin/env bash
set -euo pipefail
bench_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$bench_dir/../../.." && pwd)
# shellcheck source=/dev/null
source "$repo_dir/h800conda-tmp.sh"
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/tmp/loop-autoresearch-20261023/triton
"$PIPELINE_PYTHON" - "$repo_dir/outputs/loop-autoresearch-20261023/latest.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
assert report["correctness_failures"] == [], report["correctness_failures"]
assert len(report["round_seconds"]) == 3
assert len(report["observations"]) == 16
print("All real image-text capture guards passed.")
PY
cd "$repo_dir/vllm"
"$PIPELINE_PYTHON" -m pytest -q tests/kernels/test_sensenova_loop_mhc.py tests/test_config.py -k 'mhc or loop'
cd "$repo_dir"
"$PIPELINE_PYTHON" -m pytest -q tests/test_loop_capture.py tests/test_vllm_engine_decode.py
