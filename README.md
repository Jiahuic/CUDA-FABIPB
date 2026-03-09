# fmm_PB

Finite-memory fast multipole solver for the Poisson-Boltzmann equation using a Galerkin formulation. The current repository contains a CPU build and is being reorganized toward a cleaner `src/` and `include/` layout that can support later GPU work.

## Repository layout

- `src/`: solver source files
- `include/`: shared headers
- `build/`: generated object files
- `test_proteins/`: sample input cases
- `docs/`: project documentation

## Dependencies

This project should treat numerical libraries as system dependencies, not as vendored build scripts.

Required:

- C compiler (`gcc` or `clang`)
- BLAS
- LAPACK

Install notes are in [`docs/dependencies.md`](/Users/jiahuic/Garage/electrostatics/fmm_PB/docs/dependencies.md).

## Build

Recommended build layout:

- `build/`: CPU executable
- `build-cuda/`: CUDA-enabled executable
- `build-prof/`: profiling executable

CPU configure:

```sh
cmake -S . -B build
```

CPU build:

```sh
cmake --build build
```

CUDA configure:

```sh
cmake -S . -B build-cuda -DFMM_PB_ENABLE_CUDA=ON
```

CUDA build:

```sh
cmake --build build-cuda
```

For cross-machine performance comparisons, use a consistent benchmark setup:

```sh
./scripts/run_apples_to_apples.sh test_proteins/1a7m
```

This script enforces:
- `Release` build
- `OpenBLAS` selection (`-DFMM_PB_BLA_VENDOR=OpenBLAS`)
- single-thread runtime

### BLAS/OpenMP environment

This solver is very sensitive to BLAS/OpenMP thread settings during benchmarking.

If these are left unset, CPU and GPU comparisons can become misleading because
the GMRES path uses many BLAS vector operations and the threaded runtime
overhead can dominate the measured wall time.

For reproducible timings, export:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Use the same environment for both CPU and GPU runs.

Recommended manual comparison workflow:

```sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

./build/coulomb test_proteins/1a63
./build/coulomb -m=0 test_proteins/1a63
./build-cuda/coulomb -g=1 -m=0 test_proteins/1a63
```

Notes:

- The first run generates the mesh files.
- The `-m=0` runs reuse `.vert/.face` so remeshing does not pollute the timing.
- For scripted CPU/GPU comparisons, use:

```sh
./scripts/compare_gpu_cpu.sh test_proteins/1a63
```

Optional one-shot debug comparison of CPU vs GPU `applyFMM` on the same input:

```sh
./build-cuda/coulomb -g=1 -c=1 -m=0 test_proteins/1a63
```

This prints a single `applyFMM debug compare` line with `max_abs` and `rel_l2`
for the CPU and GPU matvec outputs before GMRES starts.

If `OpenBLAS` is unavailable, install it first (see `docs/dependencies.md`).

Clean rebuild:

```sh
rm -rf build
cmake -S . -B build
cmake --build build
```

The configure step checks BLAS and LAPACK up front and stops immediately if either is missing.

If BLAS/LAPACK live in non-default locations, pass the usual CMake search hints, for example through `CMAKE_PREFIX_PATH`.

At runtime, pass `-g=1` to request the GPU near-field backend. If the backend is unavailable or not yet fully implemented, the solver falls back to the CPU path.

## Run

CPU:

```sh
./build/coulomb test_proteins/1a7m
```

GPU:

```sh
./build-cuda/coulomb -g=1 test_proteins/1a7m
```

## Professionalization goals

The immediate cleanup target is:

- keep generated artifacts out of git
- make external dependencies explicit and overridable
- document platform setup cleanly
- preserve a stable CPU validation path while the project structure evolves
