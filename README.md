# FABIPB

FABIPB is a Galerkin boundary-integral Poisson-Boltzmann solver with FMM acceleration and an optional CUDA backend. The CPU path is kept buildable for development and laptop validation; production capsid benchmarks should run on a CUDA machine.

## Layout

- `src/`: solver implementation
- `include/`: shared headers
- `scripts/`: build, smoke-test, and benchmark helpers
- `test_proteins/`: small sample inputs and local benchmark inputs
- `docs/`: design notes and dependency details
- `results/`: generated output, ignored by git

## Dependencies

Required:

- CMake
- C compiler
- BLAS
- LAPACK

Optional:

- CUDA toolkit and NVIDIA GPU
- NanoShaper for PQR-to-surface runs

See `docs/dependencies.md` for platform-specific notes.

## Build

Default build, using CUDA automatically when CMake finds a CUDA toolkit:

```sh
cmake -S . -B build
cmake --build build -j
```

CPU-only build:

```sh
cmake -S . -B build -DFMM_PB_ENABLE_CUDA=OFF
cmake --build build -j
```

On macOS without an NVIDIA GPU, use the CPU-only mode. If using Apple Accelerate:

```sh
cmake -S . -B build -DFMM_PB_ENABLE_CUDA=OFF -DFMM_PB_BLA_VENDOR=Apple
cmake --build build -j
```

## macOS Smoke Test

Run a small CPU-only test on macOS:

```sh
./scripts/run_macos_cpu_smoke.sh
```

This configures `build-macos-cpu/`, disables CUDA, and runs a short CPU solve on `test_proteins/1bpi`, which has a checked-in mesh.

## Run

Small CPU run:

```sh
./build/fabipb -g=0 test_proteins/1bpi
```

Default run, using GPU automatically if the binary has CUDA support and a GPU is available:

```sh
./build/fabipb test_proteins/1a63.pqr
```

Show options:

```sh
./build/fabipb -h
```

## Capsid Benchmarks

ZIKV/6CO8:

```sh
SDENS=1 OUT_DIR=results/fmm/6co8_sdens1_default ./scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

H1N1:

```sh
SDENS=0.5 OUT_DIR=results/fmm/h1n1_sdens05_default ./scripts/run_6co8_fabipb_fast.sh test_proteins/H1N1_atoms.pqr
```

The runner records command, configuration, `summary.csv`, RHS diagnostics, GMRES status, energy, and timing under `results/`.

## Notes

- `results/` and local build directories are ignored by git.
- Use `-g=0` to force CPU.
- Use `-g=1` to request GPU.
- `FMM_PB_ENABLE_CUDA=OFF` disables CUDA at build time.
- `FMM_PB_BLA_VENDOR=Apple` is recommended for simple macOS builds.
- Production paper tables should be generated under `results/paper/`, which remains local-only.
