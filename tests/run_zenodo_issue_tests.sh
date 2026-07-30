#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

"$repo_dir/tests/test_pqr_parser_zero_radius.sh"
"$repo_dir/tests/test_rhs_guard.sh"
"$repo_dir/tests/test_rhs_tree_accuracy.sh"
"$repo_dir/tests/test_gpu_request_smoke.sh"
