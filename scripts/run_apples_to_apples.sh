#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"

cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFMM_PB_BLA_VENDOR=OpenBLAS

cmake --build "$BUILD_DIR"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/1a7m" >&2
  exit 2
fi

panel="$1"
shift
case "$panel" in
  *.pqr) panel="${panel%.pqr}" ;;
esac

exec ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" "$panel" "$@"
