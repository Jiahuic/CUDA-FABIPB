# OpenBLAS Sensitivity

This note records an important reproducibility issue found during cross-machine
testing.

## Observation

On one machine, the same solver/input/mesh combination behaved differently
under different OpenBLAS versions:

- OpenBLAS `0.3.8` pthread build: converged
- OpenBLAS `0.3.20` pthread build: did not converge under the same mesh-control settings
- OpenBLAS `0.3.21` pthread build: converged under the same mesh-control settings

The first divergence was traced to the LU-based preconditioner block solves.
The raw preconditioner block matrices matched before factorization, while the
`dgetrf`/`dgetrs` results did not. This means the immediate split is not in the
FMM operator, although the underlying issue is still numerical fragility in the
preconditioner blocks.

## Required Comparison Rules

- keep the exact same input, mesher, and mesh-resolution control
- use the same `-m` and `-R` values for both BLAS runs
- keep runtime thread settings pinned to `1`
- compare only one BLAS change at a time

## Recommended Workflow

Use:

```sh
./scripts/compare_openblas_runs.sh   /path/to/openblas-0.3.8/lib   /path/to/openblas-0.3.21/lib   test_proteins/1a63 -g=0 -P=2 -m=1 -R=1.0
```

If you need to reproduce the unstable environment explicitly, compare against
`0.3.20` in a separate run:

```sh
./scripts/compare_openblas_runs.sh   /path/to/openblas-0.3.20/lib   /path/to/openblas-0.3.21/lib   test_proteins/1a63 -g=0 -P=2 -m=1 -R=1.0
```

The script:

- switches `LD_LIBRARY_PATH`
- enables GMRES residual logging
- writes two full logs
- allows direct residual-history diffing

## Build-System Guard

CMake now blocks OpenBLAS `0.3.20` by default when it can resolve the linked
OpenBLAS shared library to a `0.3.20` runtime.

Override only for controlled comparison work:

```sh
cmake -S . -B build -DFMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320=ON
```

For regular development and benchmarking, prefer `0.3.21` or newer.

## Interpretation

If two OpenBLAS versions diverge under the same mesh-control settings:

- the issue should be treated as environment sensitivity first
- do not merge numerical-performance conclusions without recording the BLAS
  version
- use residual-history comparison to identify where the solver trajectories
  begin to separate
