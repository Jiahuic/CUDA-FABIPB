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

Required to build the solver:

- CMake 3.16 or newer
- C99 compiler (`gcc`, `clang`, or Apple Clang)
- BLAS
- LAPACK
- POSIX threads

Optional runtime/build features:

- CUDA toolkit and NVIDIA GPU for the GPU backend
- NanoShaper for PQR-to-surface meshing
- Python 3 for helper scripts that generate benchmark tables or notes

Debian/Ubuntu packages:

```sh
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev
```

CUDA builds also need an NVIDIA driver and CUDA toolkit with `nvcc` available:

```sh
nvcc --version
nvidia-smi
```

macOS CPU-only packages:

```sh
xcode-select --install
brew install cmake
```

Apple Accelerate can provide BLAS/LAPACK on macOS. Homebrew OpenBLAS is also
supported, but use one BLAS consistently when comparing timings.

See `docs/dependencies.md` for platform-specific notes.

## Build

Default build. CMake uses CUDA automatically when a CUDA toolkit is available:

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

Check the binary:

```sh
./build/fabipb -h
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

The benchmark runner sets the solver's production defaults automatically:
GPU selection, Q2M policy, preconditioner policy, right-hand-side tree policy,
energy mode, and thread counts. For normal runs, set only `SDENS`, `OUT_DIR`,
and the input file.

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
