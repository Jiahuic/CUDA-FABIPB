# Dependencies

This project depends on external numerical libraries and should be built against installed system packages rather than ad hoc local install scripts.

## Required libraries

- BLAS
- LAPACK

The GMRES implementation is now native C and still uses BLAS/LAPACK for the underlying linear algebra kernels.

## macOS

If the default Accelerate-backed BLAS/LAPACK setup works, a plain build is usually enough:

```sh
make
```

If you prefer Homebrew OpenBLAS/LAPACK, install packages first and then override build variables as needed:

```sh
brew install openblas lapack
make BLAS_LIBS=-lopenblas LAPACK_LIBS=-llapack
```

For apples-to-apples benchmarking against Linux, prefer OpenBLAS and run:

```sh
brew install openblas
./scripts/run_apples_to_apples.sh test_proteins/1a7m
```

## Debian/Ubuntu

```sh
sudo apt-get update
sudo apt-get install build-essential libopenblas-dev liblapack-dev
```

Configure and build with:

```sh
cmake -S . -B build
cmake --build build
```

If CMake stops because it detected OpenBLAS `0.3.20`, install or build a newer
local OpenBLAS and prefer that runtime instead of forcing the unstable one.
For example, with `gfortran` installed:

```sh
curl -L -o /tmp/OpenBLAS-0.3.31.tar.gz   https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.31.tar.gz
tar -xzf /tmp/OpenBLAS-0.3.31.tar.gz -C /tmp
cd /tmp/OpenBLAS-0.3.31
make -j4 PREFIX=/tmp/openblas-0.3.31-install   NO_SHARED=0 USE_OPENMP=0 USE_THREAD=1 DYNAMIC_ARCH=1
make PREFIX=/tmp/openblas-0.3.31-install install
```

Then run `fabipb` against that local runtime:

```sh
env LD_LIBRARY_PATH=/tmp/openblas-0.3.31-install/lib   FABIPB_SETUP_THREADS=1 FABIPB_PRECOND_APPLY_THREADS=1   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   ./build/fabipb -B=1 -g=0 -P=2 -m=0 test_proteins/1a63
```

For apples-to-apples benchmarking against macOS, run:

```sh
./scripts/run_apples_to_apples.sh test_proteins/1a7m
```

## Fedora/RHEL

```sh
sudo dnf install gcc blas-devel lapack-devel
```

Configure and build with:

```sh
cmake -S . -B build
cmake --build build
```

## Verifying linkage

Inspect detected libraries during CMake configure:

```sh
cmake -S . -B build
```

Then build:

```sh
cmake --build build
```

## Cross-machine timing checklist (Linux vs macOS)

Use the same build settings and runtime environment before comparing `ttl time`.

1. Configure and build with explicit settings:

```sh
cmake -S . -B build-a2a -DCMAKE_BUILD_TYPE=Release -DFMM_PB_BLA_VENDOR=OpenBLAS
cmake --build build-a2a
```

2. Confirm linked BLAS/LAPACK implementation:

```sh
# Linux
ldd build-a2a/fabipb | rg -i "openblas|blas|lapack"

# macOS
otool -L build-a2a/fabipb | rg -i "openblas|blas|lapack|accelerate|veclib"
```

3. Confirm thread pinning environment:

```sh
env | rg -i "OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|VECLIB_MAXIMUM_THREADS|BLIS_NUM_THREADS"
```

4. Record compiler and CMake versions:

```sh
cc --version
cmake --version
```

5. Run the standardized benchmark entrypoint:

```sh
./scripts/run_apples_to_apples.sh test_proteins/1a7m
```

Compare all five items above across machines before interpreting timing deltas.
