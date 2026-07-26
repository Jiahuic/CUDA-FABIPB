#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

set +e
output=$(
  cd "$repo_dir"
  FABIPB_MAX_DIRECT_RHS_PAIRS=1000 \
    ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=8.0 test_proteins/1a63 2>&1
)
status=$?
set -e

printf '%s\n' "$output"

if [ "$status" -eq 0 ]; then
  echo "Expected direct setupRHS guard to fail with a low pair limit" >&2
  exit 1
fi

printf '%s\n' "$output" | grep -q "direct setupRHS would require 379960 panel-charge interactions"
printf '%s\n' "$output" | grep -q "FABIPB_MAX_DIRECT_RHS_PAIRS=1000"
