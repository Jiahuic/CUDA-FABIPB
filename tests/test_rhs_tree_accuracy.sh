#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
theta=${FABIPB_TEST_RHS_TREE_THETA:-0.2}

output=$(
  cd "$repo_dir"
  FABIPB_FORCE_TREE_RHS=1 \
  FABIPB_DEBUG_COMPARE_RHS=1 \
  FABIPB_STOP_AFTER_RHS=1 \
  FABIPB_RHS_TREE_THETA="$theta" \
    ./scripts/with_benchmark_env.sh ./build/fabipb \
      -B=1 -g=0 -m=2 -R=8.0 test_proteins/1a63 2>&1
)

printf '%s\n' "$output"
printf '%s\n' "$output" | grep -q "mode=tree theta=$theta"
printf '%s\n' "$output" | grep -q "setupRHS debug compare (y0/potential):"
printf '%s\n' "$output" | grep -q "setupRHS debug compare (y1/normal-deriv):"

potential_rel_l2=$(
  printf '%s\n' "$output" |
    awk '/setupRHS debug compare \(y0\/potential\):/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^rel_l2=/) {
          split($i, value, "=")
          print value[2]
        }
      }
    }'
)
normal_rel_l2=$(
  printf '%s\n' "$output" |
    awk '/setupRHS debug compare \(y1\/normal-deriv\):/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^rel_l2=/) {
          split($i, value, "=")
          print value[2]
        }
      }
    }'
)

awk -v potential="$potential_rel_l2" -v normal="$normal_rel_l2" \
  'BEGIN { exit !(potential <= 1.0e-3 && normal <= 1.0e-3) }'
