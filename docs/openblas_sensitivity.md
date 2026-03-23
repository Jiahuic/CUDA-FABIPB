# OpenBLAS Sensitivity

This note records an important reproducibility issue found during cross-machine
testing.

## Observation

On one machine, the same solver/input/mesh combination behaved differently
under different OpenBLAS versions:

- OpenBLAS `0.3.8` pthread build: converged
- OpenBLAS `0.3.20` pthread build: did not converge on the same fixed mesh

This means benchmark and convergence results should not be treated as BLAS
version independent without verification.

## Required Comparison Rules

- keep the exact same input and generated mesh
- reuse the mesh with `-m=0`
- keep runtime thread settings pinned to `1`
- compare only one BLAS change at a time

## Recommended Workflow

Use:

```sh
./scripts/compare_openblas_runs.sh \
  /path/to/openblas-0.3.8/lib \
  /path/to/openblas-0.3.20/lib \
  test_proteins/1a63 -g=0 -P=0 -m=0
```

The script:

- switches `LD_LIBRARY_PATH`
- enables GMRES residual logging
- writes two full logs
- allows direct residual-history diffing

## Local 0.3.20 Build

On machines where the package manager only provides `0.3.8`, build `0.3.20`
locally instead of replacing the system BLAS.

Example local install target:

```sh
/tmp/openblas-0.3.20
```

Then compare by prefixing `LD_LIBRARY_PATH` instead of reinstalling system
packages.

## Interpretation

If two OpenBLAS versions diverge on the same fixed mesh:

- the issue should be treated as environment sensitivity first
- do not merge numerical-performance conclusions without recording the BLAS
  version
- use residual-history comparison to identify where the solver trajectories
  begin to separate
