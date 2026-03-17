# Benchmark Plan

This document defines the final benchmark matrix for the publication version.
The goal is to compare:

- CPU non-parallel baseline
- GPU non-parallel "full GPU-enabled" baseline
- best hybrid runtime policy

Use two benchmark drivers:

- `scripts/run_benchmark_matrix.sh`
  - depth-focused production benchmark matrix
- `scripts/run_fmm_param_matrix.sh`
  - structural and order-policy sweep for adaptive-parameter studies

All benchmark runs should use benchmark output mode:

```sh
-B=1
```

All benchmark runs should also use the benchmark wrapper so BLAS/OpenMP thread
settings stay fixed:

```sh
./scripts/with_benchmark_env.sh ...
```

## Configurations

### 1. CPU serial baseline

Purpose:

- reference accuracy and wall time
- no GPU acceleration
- no setup-thread acceleration

Settings:

- `-g=0`
- `-Q=0`
- `FABIPB_SETUP_THREADS=1`

Interpretation:

- pure CPU path for RHS, FMM apply, and setup

### 2. GPU full baseline

Purpose:

- turn on every currently implemented GPU stage
- study whether "more GPU" is always better

Settings:

- `-g=1`
- `-Q=1`
- `FABIPB_SETUP_THREADS=1`

Interpretation:

- GPU RHS
- GPU Q2M
- GPU M2L
- GPU L2P
- GPU grouped near-field
- serial setup threading for fairness against CPU serial

### 3. Hybrid best baseline

Purpose:

- best practical production configuration
- use GPU only where it currently pays off

Settings:

- `-g=1`
- `-Q=0`
- `FABIPB_SETUP_THREADS=<hybrid threads>`

Default recommendation:

- `FABIPB_SETUP_THREADS=8`

Interpretation:

- GPU RHS
- CPU Q2M
- GPU M2L
- GPU L2P
- GPU grouped near-field
- parallel preconditioner setup

This is the expected "best runtime" configuration on the current code base.

## Benchmark dimensions

For each case, benchmark across selected tree depths, for example:

- `5`
- `6`
- `7`
- `8`

For structural sweeps beyond depth-only, also vary:

- `-H=<lev>` for FMM height
- `-S=<val>` for separation ratio
- `-p=<val>` / `-pm=<val>` for order-policy sweeps

For each `(case, depth, configuration)` run, record:

- total runtime
- GMRES iteration count
- solvation energy

## Stage-level comparisons

### Top-level setup/solve

Always record:

- `loadPanel`
- `gkInit`
- `setupFMM`
- `setupPC`
- `setupRHS`
- `gmres`
- `treecode`

This answers:

- where one-time setup dominates
- whether setup work outweighs GPU matvec benefits at larger depths

### setupFMM breakdown

Always record:

- `leaf-transforms`
- `cube-alloc`
- `layouts`

And layout sub-breakdown:

- `apply`
- `panel-index`
- `cubes`
- `m2l-pairs`
- `m2l-groups`

This answers:

- whether tree/setup cost is structural or implementation-related
- which depth regime is setup-bound

### GMRES breakdown

Always record:

- `matvec`
- `psolve`
- `basis`
- `update`
- `residual`
- `other`

This answers:

- whether improvements still matter inside the iterative solve
- whether preconditioner work remains visible

### Preconditioner breakdown

Always record:

- `assemble`
- `factor`
- `solve`
- `scatter`
- `other`

This documents the effect of cached-block and cached-LU preconditioning.

### FMM stage totals

Always record:

- `Q2M`
- `M2M`
- `M2L`
- `L2L`
- `L2P`
- `Near`

This is the key stage-level comparison across CPU, GPU-full, and hybrid.

Interpretation:

- `Near` dominates in denser/local regimes
- `M2L` dominates in deeper far-field regimes
- `Q2M` is regime-dependent and should be compared explicitly between
  `gpu_full` and `hybrid_best`

### GPU near-field breakdown

For GPU-capable runs, also record:

- `build`
- `h2d`
- `kernel`
- `d2h`

And near-field build sub-breakdown:

- `meta`
- `coeff`
- `upload`
- `other`

This supports the paper narrative for the destination-grouped near-field path.

### 4. Direct-sum appendix

Purpose:

- answer the simple GPU-versus-CPU direct-sum question separately from the FMM
  study
- separate the algorithmic FMM gain from the hardware GPU gain

Settings:

- CPU direct:
  - `-g=0 -r=2`
- GPU direct:
  - `-g=1 -r=1`
- compare against:
  - CPU FMM: `-g=0 -r=0`
  - hybrid FMM: `-g=1 -Q=0 -r=0`

Interpretation:

- `cpu-direct -> gpu-direct`
  - GPU speedup for the simple direct algorithm
- `gpu-direct -> hybrid-fmm`
  - algorithmic gain from FMM on GPU
- `cpu-direct -> hybrid-fmm`
  - combined gain

Use [`scripts/compare_direct_gpu.sh`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/scripts/compare_direct_gpu.sh)
for this appendix study.

## Recommended cases

Minimum set:

- small: `test_proteins/1ajj`
- medium: `test_proteins/1a63`
- high-density medium/large:
  - regenerate with `-d=10`

Optional direct appendix:

- smaller cases using [`scripts/compare_direct_gpu.sh`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/scripts/compare_direct_gpu.sh)
- direct GPU appendix from `scripts/run_benchmark_matrix.sh`
  - this should be run once per case, not once per depth
  - it is a dense direct-sum PB reference for comparison against the FMM matrix
  - it may fall back if the dense cache does not fit GPU memory

## Output format

The benchmark driver should write:

- raw repeat CSV: one row per `(case, depth, config, repeat)`
- raw CSV: one row per `(case, depth, config)`
- summary CSV: speedups relative to CPU serial at each depth
- direct raw CSV: one row per direct-GPU repeat
- direct CSV: averaged direct-GPU appendix row for the case

Current default policy in the benchmark runner:

- `REPEATS=10`
- `DIRECT_APPENDIX=1`
- `DIRECT_DEPTH=5`
- `results_raw.csv` stores every repeat
- `results.csv` stores the averaged row for each `(case, depth, config)`
- `summary.csv` stores averaged speedups derived from those averaged rows
- `direct_results_raw.csv` stores every direct appendix repeat
- `direct_results.csv` stores the averaged direct appendix result

This keeps the data easy to analyze in Python/R/Excel and easy to reuse for
paper tables and figures.

For structural studies, the benchmark driver may later be extended so the CSV
keys include:

- `height`
- `SepRat`

## Acceptance checks

For all benchmark rows:

- same or intentionally explained `gmres-its`
- no meaningful solvation energy drift
- no unintended fallback warnings in final paper data

For correctness validation:

- keep using `-c=1` and `-C=1` separately
- do not use compare-mode runs for benchmark tables
