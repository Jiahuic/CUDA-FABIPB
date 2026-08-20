# fmm_PB

Finite-memory fast multipole solver for the Poisson-Boltzmann equation using a Galerkin formulation. The preferred workflow now uses a single build in `build/`: if a CUDA toolkit is available at configure time, the binary includes the CUDA backend and uses the GPU automatically at runtime when available.

## Repository layout

- `src/`: solver source files
- `include/`: shared headers
- `build/`: generated object files and the executable
- `results/`: generated benchmark, comparison, and calibration output
- `test_proteins/`: sample input cases
- `scripts/`: build, benchmark, and comparison helpers
- `docs/`: project documentation

## Dependencies

This project should treat numerical libraries as system dependencies, not as vendored build scripts.

Required:

- C compiler (`gcc` or `clang`)
- BLAS
- LAPACK

Install notes are in [`docs/dependencies.md`](/Users/jiahuic/Garage/electrostatics/fmm_PB/docs/dependencies.md).

## Build

Use `build/` as the single local build tree. Reconfigure that directory when
switching build types instead of keeping multiple persistent build directories.

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
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

However, some BLAS/OpenMP runtimes may read these settings before `main()`.
That means late in-process defaults are not a reliable substitute for exporting
the environment before startup. For explicit, reproducible benchmark logs,
prefer:

```sh
./scripts/with_benchmark_env.sh <command> ...
```

The benchmark wrapper also pins the solver's own worker-thread controls to `1`
unless you override them:

```sh
FABIPB_SETUP_THREADS
FABIPB_PRECOND_APPLY_THREADS
FABIPB_NEARFIELD_BUILD_THREADS
FABIPB_DIRECT_THREADS
```

Recommended manual comparison workflow:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -m=1 -R=1.0 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=1 -R=1.0 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -m=1 -R=1.0 test_proteins/1a63
```

Notes:

- Release-style runs remesh every invocation.
- Keep `-m` and `-R` fixed across CPU/GPU comparisons.
- `-R=<A>` is the backend-neutral mesh control:
  - MSMS uses `density = 1 / R^2`
  - NanoShaper uses `Grid_scale = 1 / R`
- For scripted CPU/GPU comparisons, use:

```sh
./scripts/compare_gpu_cpu.sh test_proteins/1a63
```

Optional one-shot debug comparison of CPU vs GPU `applyFMM` on the same input:

```sh
./build/fabipb -g=1 -c=1 -m=1 -R=1.0 test_proteins/1a63
```

This prints a single `applyFMM debug compare` line with `max_abs` and `rel_l2`
for the CPU and GPU matvec outputs before GMRES starts.

If `OpenBLAS` is unavailable, install it first (see `docs/dependencies.md`).

### OpenBLAS Version Sensitivity

We have observed that solver convergence can depend on the OpenBLAS version on
some machines. In the tested Linux environment behind this repository work:

- OpenBLAS `0.3.8` pthread build: converged
- OpenBLAS `0.3.20` pthread build: diverged under the same mesh-control settings
- OpenBLAS `0.3.21` pthread build: converged again under the same mesh-control settings

The first divergence was traced to the cached-LU preconditioner solve path, not
to the FMM matvec itself. Because `0.3.20` has shown unstable behavior here,
CMake now blocks it by default unless you pass:

```sh
-DFMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320=ON
```

Use that override only for controlled comparison work.

For reproducible reports:

- record the exact OpenBLAS version when reporting benchmark or convergence data
- compare BLAS versions with the same mesher and the same `-R`
- prefer switching `LD_LIBRARY_PATH` to a locally built OpenBLAS for comparison,
  rather than replacing the system BLAS install

Use the dedicated comparison runner:

```sh
./scripts/compare_openblas_runs.sh   /usr/lib/x86_64-linux-gnu/openblas-pthread   /tmp/openblas-0.3.21-runtime/usr/lib/x86_64-linux-gnu/openblas-pthread   test_proteins/1a63 -g=0 -P=2 -m=1 -R=1.0
```

This runner enables GMRES residual logging and writes side-by-side logs so the
iteration history can be diffed directly.

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

Benchmark/profiling run:

```sh
./build/fabipb -B=1 -g=1 -m=1 -R=1.0 test_proteins/1a63
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
./build/fabipb -g=1 -r=1 -m=1 -R=1.0 test_proteins/1ajj
```

Show command-line help:

```sh
./build/fabipb -h
```

### Calculation Workflows

For the full Zenodo 6CO8 calculation, use the production runner. It records
the command, mesh metadata, RHS summary, GMRES status, and stage timings under
`results/`.

`SDENS` is the user-facing NanoShaper/TABI density label. The runner translates
it internally to FABIPB's mesh parameter.

The source binary auto-selects GPU mode, Q2M/L2P placement, preconditioner,
worker thread counts, and panel-tree energy defaults. The runner handles mesh
setup, output directories, tree RHS, and result summaries. Expert overrides
such as `FMM_GPU`, `FMM_Q2M`, and `FMM_PRECONDITIONER` are still available but
are not needed for normal capsid runs.

```sh
SDENS=1 \
OUT_DIR=results/fmm/6co8_fabipb_sdens1/zenodo_sdens1_depth8_full \
./scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr

SDENS=2 FMM_DEPTH=9 \
OUT_DIR=results/fmm/6co8_fabipb_sdens2/zenodo_sdens2_depth9_full \
./scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

H1N1 at the paper density:

```sh
SDENS=0.5 \
OUT_DIR=results/fmm/h1n1_fabipb_sdens05/h1n1_sdens05_depth8_full \
./scripts/run_6co8_fabipb_fast.sh test_proteins/H1N1_atoms.pqr
```

Run the supported regression checks after rebuilding:

```sh
cmake --build build -j 4
./tests/run_zenodo_issue_tests.sh
```

For matched TABI-PB/FABIPB size comparisons:

```sh
./scripts/compare_tabi_fmm_size_sweep.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

For CPU/GPU FMM comparisons on a small or medium protein:

```sh
./scripts/compare_gpu_cpu.sh test_proteins/1a63
```

For systematic benchmark and parameter studies:

```sh
./scripts/run_benchmark_matrix.sh test_proteins/1a63
./scripts/run_fmm_param_matrix.sh test_proteins/1a63
```

The benchmark matrix already includes the `hybrid_best` configuration; the
old standalone hybrid-only runner is archived under `scripts/archive/`.

Notes:

- default output is intentionally quiet: final wall time and solvation energy
- pass `-B=1` to print setup, GMRES, FMM, and GPU cache/profiling details
- pass `-R=<A>` to choose a backend-neutral mesh resolution
- pass `-d=<val>` only for explicit backend-specific overrides
- pass `-M=1` for mesh-only calibration runs
- `-r=1` selects the direct GPU baseline matvec instead of the FMM matvec
- direct mode is intended for benchmark/reference use and may be limited by GPU memory
- direct mode prints its estimated host/device memory footprint before allocation
- compare modes should not be used for timing runs

## Mesh calibration workflow

For branch experiments that compare MSMS and NanoShaper under one shared mesh
control:

```sh
RESOLUTIONS="0.75 1.00 1.25 1.50 2.00" \
./scripts/calibrate_mesh_resolution.sh test_proteins/1ajj
```

This produces:

- `mesh_calibration.csv`
- `mesh_panel_matches.csv`

To estimate a mesher setting for a target kept-panel count:

```sh
./scripts/fit_mesh_target.sh results/mesh_calibration/.../mesh_calibration.csv msms 5000 test_proteins/1ajj
./scripts/fit_mesh_target.sh results/mesh_calibration/.../mesh_calibration.csv nanoshaper 5000 test_proteins/1ajj
```

Or ask the calibration sweep to emit fitted recommendations directly:

```sh
TARGET_KEPT_PANELS=5000 \
./scripts/calibrate_mesh_resolution.sh test_proteins/1ajj
```

To let the sweep automatically sample around the fit until the kept-panel error
falls within a tolerance:

```sh
TARGET_KEPT_PANELS=5000 \
TARGET_TOLERANCE=100 \
TARGET_MAX_ITERS=4 \
./scripts/calibrate_mesh_resolution.sh test_proteins/1ajj
```

The calibration run also emits a monotonicity report so you can see whether
kept-panel count decreases cleanly as `R` increases for each backend.
If a backend is non-monotone over the sampled points, target fitting falls back
to a conservative nearest-sample recommendation instead of interpolating.
Those fallback cases are summarized in `mesh_target_warnings.csv`.
Each calibration run also writes `mesh_calibration_health.txt` with the key
monotonicity status, final recommendations, warnings, and output file paths.

## Professionalization goals

The immediate cleanup target is:

- keep generated artifacts out of git
- make external dependencies explicit and overridable
- document platform setup cleanly
- preserve a stable CPU validation path while the project structure evolves
