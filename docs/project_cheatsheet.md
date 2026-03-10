# Project Cheat Sheet

## Current status

- Repository baseline: CPU solver works, CUDA build exists.
- Known-good git commit: `7c3371a` (`kernel-port: flatten m2l traversal and add cuda nearfield path`)
- Current intended build split:
  - `build/` = CPU executable
  - `build-cuda/` = CUDA-enabled executable
  - `build-prof/` = CPU profiling executable

## What is accelerated today

- CPU path: full solver
- CUDA path at known-good baseline:
  - near-field (`P2P`) cache + apply
  - CPU fallback for the rest

## Why the GPU version was faster at the good baseline

For the March 8, 2026 comparison on `test_proteins/1a63`:

- CPU log: [`cpu.log`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/build-cuda/compare_logs/20260308_230253/cpu.log)
  - `ttl time: 12.203079`
  - `Near=8.197617 s`
- GPU log: [`gpu.log`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/build-cuda/compare_logs/20260308_230253/gpu.log)
  - `ttl time: 5.969664`
  - `Near=2.048872 s`

Interpretation:
- The speedup came from the GPU near-field path.
- `M2L`, `Q2M`, `M2M`, `L2L`, and `L2P` were still effectively CPU-cost-sized.

## Build commands

CPU build:

```sh
cmake -S . -B build
cmake --build build
```

CUDA build:

```sh
cmake -S . -B build-cuda -DFMM_PB_ENABLE_CUDA=ON
cmake --build build-cuda
```

Profiling build:

```sh
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE='-O2 -g -pg' \
  -DCMAKE_EXE_LINKER_FLAGS='-pg'
cmake --build build-prof
```

## Run commands

CPU:

```sh
./build/fabipb test_proteins/1a63
```

GPU:

```sh
./build/fabipb -g=1 test_proteins/1a63
```

Profiling:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
rm -f gmon.out
./build-prof/fabipb -g=0 test_proteins/1a7m > build-prof/cpu_profile_run.log 2>&1
gprof ./build-prof/fabipb gmon.out > build-prof/gprof.txt
```

## Runtime prerequisites

- CUDA compilation requires `nvcc`.
- CUDA execution requires a working driver/device stack.
- Quick runtime check:

```sh
nvidia-smi
```

If `nvidia-smi` fails, the CUDA binary may still build, but runtime will fall back or fail.

## Input files and mesh notes

- Some runs need generated mesh artifacts such as `.vert` and `.face`.
- If you see:

```text
cannot open vertices file '...vert' (run msms first)
```

the solver failed before GPU performance even mattered.

## Comparison workflow

Reference comparison logs live under:

- [`build-cuda/compare_logs/20260308_230253`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/build-cuda/compare_logs/20260308_230253)

When comparing CPU vs GPU, keep:

- same input case
- same thread settings
- same build type
- same machine

## Safe rollback workflow

If an experiment becomes slower or unstable:

1. Save the diff:

```sh
git diff > /tmp/experiment.patch
```

2. Restore the known-good files from `HEAD`:

```sh
git restore --source=HEAD -- include/gpu_backend.h src/fmm.c src/gpu_backend_cuda.cu src/gpu_backend_stub.c
```

3. Rebuild:

```sh
cmake --build build
cmake --build build-cuda
```

4. If you want to revisit the experiment later, use a new branch:

```sh
git switch -c gpu-experiment-retry
git apply /tmp/experiment.patch
```

## Current hotspots

From CPU profiling on `1a7m`:

- Near-field dominates
- `M2L` is second
- `Q2M`, `M2M`, `L2L`, `L2P` are much smaller

That means:
- near-field correctness and throughput matter most
- `M2L` is the next serious target
- leaf-stage GPU work should not be allowed to hurt the already good near-field speedup

## What we tried and why it was reverted

Attempted uncommitted work:

- GPU `M2L`
- GPU `Q2M`
- GPU `L2P`
- extra flattening in `src/fmm.c`

Problem:

- the added implementation became slower and unstable
- it was not committed
- it has been preserved separately instead of left in the working tree

Saved experimental artifacts:

- `/tmp/gpu-fabipb-wip-20260309.patch`
- `/tmp/gpu_backend.h.wip`
- `/tmp/fmm.c.wip`
- `/tmp/gpu_backend_cuda.cu.wip`
- `/tmp/gpu_backend_stub.c.wip`

## Recommended next step

Do not push more FMM stages to GPU yet.

Next highest-value work:

1. Keep the known-good near-field GPU version as the baseline.
2. Add measurement around the existing GPU near-field cache build and apply path.
3. Optimize the current near-field kernel before touching `M2L` again.
4. Only return to `M2L` after adding a small correctness harness that compares CPU vs GPU local expansions on one case and one matvec.

That harness now exists as a one-shot `applyFMM` comparison mode:

```sh
./build/fabipb -g=1 -c=1 -m=0 test_proteins/1a63
```

Look for:

```text
applyFMM debug compare: max_abs=... rel_l2=...
```

## Practical rule for future work

Before changing solver stages:

- keep one known-good log pair
- keep one known-good commit
- make one GPU change at a time
- verify:
  - build success
  - same solvation energy
  - same GMRES iteration count
  - stage timings improve, not just total time
