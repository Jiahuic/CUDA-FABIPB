#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# A pair count above FABIPB_MAX_DIRECT_RHS_PAIRS no longer aborts: setupRHS
# falls back to the tree-accelerated charge-to-panel path (chargeTree.c /
# rhsTreeWalk in treecode.c) instead of erroring out.
set +e
output=$(
  cd "$repo_dir"
  FABIPB_MAX_DIRECT_RHS_PAIRS=1000 \
    ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=8.0 test_proteins/1a63 2>&1
)
status=$?
set -e

printf '%s\n' "$output"

if [ "$status" -ne 0 ]; then
  echo "Expected setupRHS to fall back to the tree path and succeed with a low pair limit" >&2
  exit 1
fi

printf '%s\n' "$output" | grep -q "379960 panel-charge interactions"
printf '%s\n' "$output" | grep -q "FABIPB_MAX_DIRECT_RHS_PAIRS=1000"
printf '%s\n' "$output" | grep -q "using tree-accelerated RHS"
printf '%s\n' "$output" | grep -q "mode=tree"

# FABIPB_ALLOW_LARGE_DIRECT_RHS=1 preserves its original meaning: force the
# direct loop even above the pair limit.
output=$(
  cd "$repo_dir"
  FABIPB_MAX_DIRECT_RHS_PAIRS=1000 FABIPB_ALLOW_LARGE_DIRECT_RHS=1 \
    ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=8.0 test_proteins/1a63 2>&1
)
printf '%s\n' "$output" | grep -q "mode=direct"
