# fmm_PB

Finite-memory fast multipole solver for the Poisson-Boltzmann equation using a Galerkin formulation. The preferred workflow now uses a single build in `build/`: if a CUDA toolkit is available at configure time, the binary includes the CUDA backend and uses the GPU automatically at runtime when available.

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

- `build/`: default executable, with CUDA enabled automatically when available
- `build-prof/`: optional profiling build

Default configure:

```sh
cmake -S . -B build
```

Default build:

```sh
cmake --build build
```

Optional explicit CPU-only configure:

```sh
cmake -S . -B build -DFMM_PB_ENABLE_CUDA=OFF
```

Optional profiling configure:

```sh
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-prof
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

The solver now defaults these common runtime variables to `1` if they are not
already set:

```sh
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
VECLIB_MAXIMUM_THREADS
BLIS_NUM_THREADS
```

This is a runtime concern, not a reliable compile-time setting across BLAS
vendors. For explicit, reproducible benchmark logs, prefer:

```sh
./scripts/with_benchmark_env.sh <command> ...
```

Recommended manual comparison workflow:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=0 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=0 test_proteins/1a63
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
./build/fabipb -g=1 -c=1 -m=0 test_proteins/1a63
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

At runtime:

- default startup uses the GPU automatically when the backend is present
- default FMM startup uses the grouped near-field GPU path when GPU is enabled
- default preconditioner is the cached-LU path (`-P=2`)
- pass `-g=0` to force CPU
- pass `-g=1` to require/request GPU explicitly

If the backend is unavailable or not yet fully implemented for a path, the solver falls back to the CPU path.

## Run

Default run (auto GPU if available):

```sh
./build/fabipb test_proteins/1a7m
```

Force CPU:

```sh
./build/fabipb -g=0 test_proteins/1a7m
```

Force GPU:

```sh
./build/fabipb -g=1 test_proteins/1a7m
```

Direct GPU dense baseline:

```sh
./build/fabipb -g=1 -r=1 -m=0 test_proteins/1ajj
```

Show command-line help:

```sh
./build/fabipb -h
```

Notes:

- `-r=1` selects the direct GPU baseline matvec instead of the FMM matvec
- direct mode is intended for benchmark/reference use and may be limited by GPU memory
- direct mode prints its estimated host/device memory footprint before allocation
- `-c=1` and `-C=1` are development-only compare modes and should not be used for timing runs

## Professionalization goals

The immediate cleanup target is:

- keep generated artifacts out of git
- make external dependencies explicit and overridable
- document platform setup cleanly
- preserve a stable CPU validation path while the project structure evolves
