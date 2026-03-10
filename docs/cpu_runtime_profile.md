# CPU Runtime Profile (Function-Level)

Date: 2026-03-08  
Case: `test_proteins/1a7m`  
Mode: CPU (`-g=0`)  
Threads: single-thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`)

## Commands used

```sh
cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE='-O2 -g -pg' \
  -DCMAKE_EXE_LINKER_FLAGS='-pg'
cmake --build build-prof

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
rm -f gmon.out
./build-prof/fabipb -g=0 test_proteins/1a7m > build-prof/cpu_profile_run.log 2>&1
gprof ./build-prof/fabipb gmon.out > build-prof/gprof.txt
```

## Overall runtime context

From solver output:
- `ttl time: 16.471177 s`
- `gmres-its: 17`
- `FMM matvec calls: 18`

Built-in FMM stage timing:
- `Near: 9.885044 s` (dominant stage)
- `M2L: 0.947075 s`
- Remaining stages (`Q2M/M2M/L2L/L2P`) are much smaller.

## Top hotspot functions (gprof flat profile)

1. `pnlNil0` - `29.15%`
2. `kernelKER4` - `17.89%`
3. `panelIA0` - `9.84%`
4. `nrCommonVtx` - `8.06%`
5. `panelRHS` - `6.81%`
6. `applyNearfield1` - `5.21%` (self time only; total impact is larger through children)
7. `convM2L` - `4.68%`
8. `pnlOne0` - `3.20%`
9. `kernelRHS` - `2.25%`
10. `setupDerivs` - `2.01%`

Note:
- `gprof` sampling has limited resolution and focuses on instrumented user-space symbols.
- BLAS/LAPACK internals are not expanded here, but FMM nearfield and `M2L` dominate regardless.

Call-graph highlights:
- `gmres -> MtVmain -> applyFMM` dominates iterative solve time.
- Inside `applyFMM`, `applyNearfield1` is the largest contributor.
- `panelIA0` and quadrature helpers (`pnlNil0/pnlOne0/pnlTwo0/...`) dominate nearfield work.

## What this means for GPU priority

Priority for acceleration should remain:
1. Nearfield `P2P` path (`applyNearfield1` + `panelIA0`/kernel math)
2. `M2L` path (`transM2L`/`convM2L`)

This matches the current GPU roadmap.

## Raw profiling artifacts

- [gprof_1a7m_cpu.txt](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/docs/profiling/gprof_1a7m_cpu.txt)
