#!/usr/bin/env sh
set -eu

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  echo "Example: $0 ./build/fabipb -m=1 -d=10 test_proteins/1a63" >&2
  exit 2
fi

echo "Benchmark env:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  VECLIB_MAXIMUM_THREADS=$VECLIB_MAXIMUM_THREADS"
echo "  BLIS_NUM_THREADS=$BLIS_NUM_THREADS"

exec "$@"
