#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

output=$(
  cd "$repo_dir"
  ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=1 -m=2 -R=8.0 test_proteins/1a63 2>&1
)

printf '%s\n' "$output"

printf '%s\n' "$output" | grep -q "GPU mode=1"
printf '%s\n' "$output" | grep -q "setupRHS direct pairs: panels=184 charges=2065 pairs=379960"
printf '%s\n' "$output" | grep -Eq "CUDA backend unavailable|GPU RHS cache|GPU backend requested"
printf '%s\n' "$output" | grep -q "solvation energy:"
