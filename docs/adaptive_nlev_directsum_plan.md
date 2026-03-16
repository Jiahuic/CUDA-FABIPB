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

## Full FMM parameter set

The adaptive-depth study should be based on the actual FMM control parameters in
this code, not just on `nLev`.

### Structural parameters

- `depth` (`-t=<lev>`)
  - finest tree depth
  - the main `nLev` parameter
- `height` (currently fixed internally at `2`)
  - coarsest active FMM level
  - not yet exposed on the command line
- `maxSepRatio` (`-S=<val>`)
  - separation/admissibility threshold
  - changes the balance between near-field and far-field work

### Expansion parameters

- `order` (`-p=<val>`)
  - controls the FMM order policy
  - if positive: fixed order on all active levels
  - if negative: variable order with `-order` used at the finest level
- `orderMom` (`-pm=<val>`)
  - override for moment/local orders relative to `M2L`
  - only meaningful when variable-order mode is used

### Derived FMM parameters

These are computed in `gkInit()` from the structural and expansion parameters:

- `ordM2L[lev]`
  - `M2L` order used at each level
- `ordMom[lev]`
  - moment/local order used for `Q2M`, `M2M`, `L2L`, and `L2P`
- `maxOrder`
  - largest order used anywhere
- `nMom[p]`
  - number of coefficients for order `p`

### Quadrature/interface parameter

- `maxQuadOrder` (`-q=<ord>`)
  - panel quadrature order
  - affects near-field, RHS, and related panel-integral work

### Execution-policy parameters

These do not change the mathematical FMM hierarchy directly, but they do change
which `nLev` is optimal once GPU is involved:

- `gpuMode` (`-g=0|1`)
- `gpuQ2MMode` (`-Q=0|1`)
- `gpuNearfieldMode` (`-G=0|1`)
- `precondCacheMode` (`-P=0|1|2`)

For the adaptive strategy, these should be treated as part of the runtime model,
not ignored as "implementation details."

## Initial study scope

The first adaptive-depth study should stay narrow.

### Vary first

- `depth`
- execution mode:
  - CPU serial
  - GPU full
  - hybrid best

### Hold fixed initially

- `height = 2`
- `maxSepRatio = 0.8`
- `qOrder = 1`
- chosen `order` / `orderMom` policy

This gives a manageable first model. If depth-only adaptation is not stable
enough, then the next parameters to widen are:

1. `maxSepRatio`
2. `order`
3. `height`

## Workstream A: direct-sum PB comparison

### Objective

Establish a direct-sum GPU reference point for PB matvecs.

### Plan

1. Use the in-tree direct GPU baseline first:
   - `fabipb -g=1 -r=1`
2. Restrict this comparison to cases that actually fit the dense direct cache.
3. Run it on the same GPU machine used for the current benchmarks.
4. Use matching or near-matching cases:
   - `1ajj`
   - one medium case only if it fits
   - do not assume `1a63` at density `10` is feasible
5. Record:
   - wall time
   - iteration count
   - energy / accuracy agreement
   - GPU hardware
6. Compare against:
   - current GPU-FMM hybrid path
7. Only if the in-tree direct path is not sufficient, retrieve and build the
   older CUDA direct-sum PB solver from the shared Dropbox location:
   - `sharing_chen/code/cuda_related`

Important implementation note:

- the current in-tree direct baseline is a dense cached operator
- for large cases it can fail by memory and fall back to FMM
- this is documented in `docs/direct_gpu_limitations.md`

### Decision rule

- If a true direct GPU PB path is much faster than our current near-field
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

In the first pass, this is a depth-only model with all other FMM parameters held
fixed. Only after that should the model be widened to include additional FMM
knobs such as `maxSepRatio` or order policy.

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

1. Use the in-tree direct GPU baseline on small cases that fit memory.
2. If needed, retrieve the older direct-sum PB solver and compare against it.
3. Extend the benchmark matrix outputs into a depth-selection study.
4. Add a report-only `nLev` recommendation mode in this repo after the direct
   comparison is complete.
