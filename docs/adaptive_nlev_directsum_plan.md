# Adaptive `nLev` and Direct-Sum Comparison Plan

## Scope

This branch is for collaborator points 1 and 2:

- update the `nLev` strategy when GPU acceleration is available
- compare the current GPU-PB code against a direct-sum GPU PB solver to judge
  whether the current near-field speedup is already near the hardware limit

The goal is to answer two related questions:

1. When GPU acceleration changes the relative cost of `Near` and `M2L`, how
   should the code choose `nLev`?
2. Is the current near-field GPU acceleration already close to what a direct
   GPU PB solver can achieve on the same machine?

## Current evidence

From the current benchmark matrix on `1a63` at density `10`:

- GPU grouped near-field gives about `19x-26x` speedup over CPU
- GPU `M2L` gives about `4x-5x` speedup over CPU
- the best overall runtime is a hybrid policy rather than a fixed "all GPU"
  policy
- as depth increases, the solve shifts from near-field-dominated to
  `M2L`-dominated

So the old serial-only `nLev` rule is no longer sufficient. The new rule needs
to account for different CPU/GPU speedups for different FMM stages.

## Workstream A: direct-sum PB comparison

### Objective

Establish an external GPU direct-sum reference point.

### Plan

1. Retrieve and build the older CUDA direct-sum PB solver from the shared
   Dropbox location:
   - `sharing_chen/code/cuda_related`
2. Run it on the same GPU machine used for the current benchmarks.
3. Use matching or near-matching cases:
   - `1ajj`
   - `1a63`
   - possibly one density-10 case if feasible
4. Record:
   - wall time
   - iteration count if iterative
   - energy / accuracy agreement if available
   - GPU hardware
5. Compare against:
   - current direct GPU baseline in this repo (`-r=1`) where memory allows
   - current GPU-FMM hybrid path

### Decision rule

- If the older direct-sum PB solver is much faster than our current near-field
  behavior would suggest, then the near-field CUDA path still needs optimization.
- If the speedup is in the same regime as the current `19x-26x`, then the
  current near-field implementation is likely reasonable and the emphasis should
  shift to adaptive `nLev` and final benchmark production.

## Workstream B: adaptive `nLev` model

### Objective

Replace the serial-only `nLev` strategy with a GPU-aware hybrid strategy.

### Principle

The optimal depth should balance:

- near-field work
- far-field `M2L` work
- setup cost
- GPU/CPU stage-specific speedups

The current data already suggests:

- smaller `nLev` means larger leaves and more direct/near work
- larger `nLev` means more boxes and more `M2L` work
- GPU changes this tradeoff because `Near` and `M2L` accelerate differently

### Phase 1: empirical model

Use the benchmark matrix to fit a practical rule from measured stage costs.

For each `(case, density, depth)` collect:

- `Near`
- `M2L`
- `Q2M`
- `L2P`
- `setupFMM`
- total runtime

for:

- CPU serial
- GPU full
- hybrid best

Then fit a simple runtime estimate such as:

`T(depth) = T_setup(depth) + a * Near(depth) + b * M2L(depth) + c * other(depth)`

where `a`, `b`, and `c` depend on the execution mode:

- CPU serial
- GPU full
- hybrid

### Phase 2: runtime policy

Implement a nonintrusive recommendation mode first:

- input: case geometry after mesh generation
- output: recommended `nLev`

This should start as:

- report-only mode
- no automatic override of the user-specified `nLev`

Once the heuristic is stable, it can become an automatic mode.

### Phase 3: hybrid-aware rule

Extend the policy so it can also choose:

- CPU vs GPU `Q2M`
- possibly other stage-specific hybrid choices later

This is aligned with the current project direction:

- final production mode should be adaptive and hybrid
- not fixed CPU-only or GPU-only

## Deliverables

1. Benchmark comparison table against the older direct-sum PB solver.
2. A depth-vs-runtime study showing where GPU `Near` and GPU `M2L` dominate.
3. A first GPU-aware `nLev` recommendation heuristic.
4. A final paper discussion section explaining why GPU changes the classic
   serial `nLev` tradeoff.

## Immediate next steps

1. Retrieve the older direct-sum PB solver and build it on the GPU machine.
2. Run a matched comparison on `1ajj` and `1a63`.
3. Extend the benchmark matrix outputs into a depth-selection study.
4. Add a report-only `nLev` recommendation mode in this repo after the direct
   comparison is complete.
