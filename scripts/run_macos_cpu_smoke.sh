#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" && "${ALLOW_NON_MACOS:-0}" != "1" ]]; then
  echo "This smoke test is intended for macOS CPU-only builds." >&2
  echo "Set ALLOW_NON_MACOS=1 only when validating the script on another OS." >&2
  exit 1
fi

build_dir="${BUILD_DIR:-$repo_root/build-macos-cpu}"
case_path="${1:-$repo_root/test_proteins/1bpi}"
max_iter="${GMRES_MAX_ITER:-2}"

if [[ -z "${JOBS:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    jobs="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
  else
    jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
  fi
else
  jobs="$JOBS"
fi

cmake_args=(
  -S "$repo_root"
  -B "$build_dir"
  -DFMM_PB_ENABLE_CUDA=OFF
)

if [[ -n "${FMM_PB_BLA_VENDOR+x}" ]]; then
  blas_vendor="$FMM_PB_BLA_VENDOR"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  blas_vendor="Apple"
else
  blas_vendor=""
fi

if [[ -n "$blas_vendor" ]]; then
  cmake_args+=("-DFMM_PB_BLA_VENDOR=$blas_vendor")
fi

echo "[macos-smoke] configure CPU-only build: $build_dir"
cmake "${cmake_args[@]}"

echo "[macos-smoke] build"
cmake --build "$build_dir" -j "$jobs"

echo "[macos-smoke] run CPU-only small mesh: $case_path"
env \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 \
  BLIS_NUM_THREADS=1 \
  "$build_dir/fabipb" \
    -B=1 -g=0 -m=1 -R=1 -t=4 -P=3 -a=10 -i="$max_iter" -o=1e-3 \
    "$case_path"

echo "[macos-smoke] ok"
