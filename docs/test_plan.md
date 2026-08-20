# Test Plan

This document defines the validation and benchmark workflow for the current
publication baseline on `main`.

Current baseline includes:

- grouped GPU near-field
- grouped GPU `M2L`
- GPU `Q2M` / `L2P`
- cached-LU preconditioner
- parallel preconditioner setup
- serial `setupRHS`

## 1. Build Matrix

Default build:

```sh
cmake -S . -B build
cmake --build build
```

Profiling build:

```sh
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-prof
```

## 2. Smoke Tests

Small case:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=1 -R=1.0 test_proteins/1ajj
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -R=1.0 test_proteins/1ajj
```

Pass criteria:

- same `gmres-its`
- same solvation energy to printed precision
- no unexpected GPU fallback warnings

## 3. Numerical Correctness Tests

Use compare modes only for validation, not timing.

ApplyFMM compare:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -c=1 -m=1 -R=1.0 test_proteins/1ajj
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -c=1 -m=1 -R=1.0 test_proteins/1a63
```

Preconditioner compare:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -C=1 -m=1 -R=1.0 test_proteins/1ajj
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -C=1 -m=1 -R=1.0 test_proteins/1a63
```

Pass criteria:

- `applyFMM debug compare` near machine precision
- `PtVfmm debug compare` near machine precision

## 4. CPU vs GPU Benchmark Runs

These are the standard timing runs for the solver:

```sh
./scripts/compare_gpu_cpu.sh test_proteins/1ajj
./scripts/compare_gpu_cpu.sh test_proteins/1a63
```

Record:

- `ttl time`
- `gmres-its`
- solvation energy
- top-level stage times
- GMRES breakdown
- preconditioner breakdown
- FMM stage totals

## 5. Direct GPU Baseline

Use the dense direct GPU baseline for the paper comparison story.

Small/medium case first:

```sh
./scripts/compare_direct_gpu.sh test_proteins/1ajj
```

Record:

- CPU baseline
- direct GPU baseline
- GPU-FMM baseline

If direct GPU fails because of memory, record that explicitly.

## 6. High-Density Mesh Cases

Use at least one denser MSMS case. Prefer the normalized control for routine
comparisons and use backend-specific override only when explicitly studying the
MSMS density knob:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -R=0.5 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=1 -R=0.5 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -d=10 test_proteins/1a63
```

Record:

- `#ele`
- wall time
- setup breakdown
- FMM breakdown
- direct-GPU feasibility if attempted

## 7. Setup Parallelism Validation

Validate the parallel preconditioner setup:

```sh
FABIPB_SETUP_THREADS=1 ./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -R=1.0 test_proteins/1a63
FABIPB_SETUP_THREADS=8 ./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -R=1.0 test_proteins/1a63
```

Record:

- `setupPC`
- `ttl time`
- same `gmres-its`
- same energy

## 8. Paper Table Structure

For each case, keep:

- case name
- `#Atoms`
- `#ele`
- CPU wall time
- GPU-FMM wall time
- direct GPU wall time if available
- CPU / GPU-FMM speedup
- direct-GPU / GPU-FMM speedup
- `gmres-its`
- energy difference
- direct-GPU memory note if it fails

## 9. Acceptance Thresholds

Use these consistently:

- no change in `gmres-its`
- no meaningful solvation energy drift
- compare-mode relative error near machine precision for validated GPU paths
- no benchmark tables generated with `-c=1` or `-C=1`

## 10. Final Freeze Checklist

Before writing final benchmark tables:

1. Rebuild from a clean `build/`.
2. Rerun one small case.
3. Rerun one medium case.
4. Rerun one high-density case.
5. Rerun one direct-GPU comparison.
6. Confirm final logs do not show unintended CPU fallbacks.

## 11. Mesh Calibration

For MSMS vs NanoShaper calibration work, use mesh-only mode:

```sh
./build/fabipb -M=1 -m=1 -R=1.0 test_proteins/1ajj
./build/fabipb -M=1 -m=2 -R=1.0 test_proteins/1ajj
```

Or sweep both backends:

```sh
RESOLUTIONS="0.75 1.00 1.25 1.50 2.00" \
./scripts/calibrate_mesh_resolution.sh test_proteins/1ajj
```
