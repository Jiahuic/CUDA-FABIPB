#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

case_name="pqr_parser_zero_radius_$$"
case_base="test_proteins/$case_name"
trap 'rm -f "$repo_dir/$case_base.pqr" "$repo_dir/$case_base.xyzr" "$repo_dir/$case_base.face" "$repo_dir/$case_base.vert"' EXIT INT TERM

cp "$repo_dir/test_proteins/1a63.pqr" "$repo_dir/$case_base.pqr"
cat >> "$repo_dir/$case_base.pqr" <<'PQR'
ATOM   2066  H   ALA A   1       1.000   1.000   1.000  0.5000 0.0000
PQR

output=$(
  cd "$repo_dir"
  ./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=2 -R=8.0 -M=1 "$case_base" 2>&1
)

printf '%s\n' "$output"

printf '%s\n' "$output" | grep -q "mesh_atoms=2065 charge_atoms=2066"
printf '%s\n' "$output" | grep -q "kept 1 zero-radius atoms as charges"
