# `panelIA0()` GPU Plan

## Objective

Reduce near-field setup time by moving part of the `panelIA0()` coefficient build off the CPU.

Current state:
- `buildNearfieldTables()` in [`src/gpu_backend_cuda.cu`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/gpu_backend_cuda.cu) expands near leaf-pairs into panel-pairs and calls `panelIA0()` on the CPU for each pair.
- `gpuNearfieldApply()` only applies the cached coefficients on the GPU.
- Prior profiling showed that near-field setup is dominated by coefficient generation, not by metadata or upload.

So the real target is:
- CPU `panelIA0()` evaluation inside near-field setup

Not the repeated apply kernel, which is already in good shape.

## What `panelIA0()` Does

`panelIA0()` in [`src/numQuad.c`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c) is not a single formula. It does:

1. Topological classification with [`nrCommonVtx()`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c)
2. Case dispatch to one of:
   - [`pnlNil0()`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c) for disjoint panels
   - [`pnlOne0()`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c) for one shared vertex
   - [`pnlTwo0()`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c) for one shared edge
   - [`pnlThr0()`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/numQuad.c) for self interaction
3. Scaling and output of 4 coefficients

This is why it is expensive:
- branch-heavy
- geometry-heavy
- singular or near-singular cases need special quadrature

## Feasibility Summary

### Feasible now

- CPU classification of panel pairs
- GPU evaluation of the disjoint case only
- CPU fallback for touching or singular cases
- Reuse of the current flattened near-field cache format

### Feasible later, but high risk

- GPU implementation of all `panelIA0()` cases
- Full device-side port of the singular quadrature paths

### Not the right first step

- Fully GPU `panelIA0()` with all cases in one kernel
- On-the-fly mixed CPU/GPU per-pair evaluation

Those are too risky compared with the current code state.

## Recommended Staged Plan

### Stage 0: Keep the Baseline

Baseline remains:
- CPU builds all near-field coefficients with `panelIA0()`
- GPU applies them in `gpuNearfieldApply()`

This stays as the validation reference.

### Stage 1: Add CPU Classification and Case Histogram

Add a preprocessing pass that:
- classifies each near panel-pair by `nrCommonVtx()`
- counts:
  - disjoint
  - one-shared-vertex
  - shared-edge
  - self

Purpose:
- quantify whether the disjoint case is large enough to justify GPU work
- avoid porting singular kernels blindly

Success criterion:
- exact match with current `panelIA0()` case logic

### Stage 2: GPU Disjoint Prototype

Implement only the disjoint path on GPU:
- CPU classifies pairs
- CPU sends only disjoint pairs to a GPU builder kernel
- GPU evaluates the equivalent of `pnlNil0()`
- CPU still handles all touching and singular cases

Purpose:
- lowest-risk prototype
- directly tests whether the dominant safe case is enough to matter

Success criterion:
- coefficient agreement for the disjoint bucket
- stable end-to-end solve
- measurable near-field setup reduction

### Stage 3: Decide Whether to Continue

After the disjoint prototype, evaluate:
- fraction of total pairs that are disjoint
- fraction of total build time attributable to disjoint pairs
- net wall-time improvement

Decision:
- if the gain is good, continue
- if the gain is small, stop and keep the CPU build path

### Stage 4: Singular Case GPU Port

Only if Stage 2 justifies it:
- port `pnlOne0()`
- port `pnlTwo0()`
- port `pnlThr0()`
- keep separate kernels or buckets by case type

This is the ambitious path, not the starting point.

## Data Layout

The GPU builder should not use `panel *` directly. Use flattened arrays derived from the current panel data.

### Panel geometry

Minimum geometry needed on device:
- area
- normal
- `vtx[3][3]`
- `a[3][3]`
- global vertex IDs

This can start as an array-of-structs and be changed later if profiling demands it.

### Pair metadata

Per pair, store:
- destination panel index
- source panel index
- case type
- `idxX[3]`
- `idxY[3]`

This metadata is small and computed once on CPU.

### Output

Reuse the current near-field coefficient format:
- `k0`
- `k1`
- `k2`
- `k3`
- source and destination panel indices as needed

That keeps `gpuNearfieldApply()` unchanged.

## Why This Split Is Reasonable

The right CPU/GPU split is:

CPU:
- mesh connectivity
- shared-vertex classification
- bucket construction
- singular fallback

GPU:
- floating-point coefficient evaluation for safe cases
- coefficient writeout in the existing cache format

This avoids:
- repeated CPU/GPU round trips
- pointer-rich device logic
- mixing topological logic with quadrature kernels

## Validation Plan

For each stage, compare against the current baseline:

- coefficient error for `k0..k3`
- near-field matvec agreement
- GMRES iteration count
- solvation energy

Test on:
- one small debug case
- one medium case
- one larger benchmark case

## Performance Measurements

Track:
- case histogram
- CPU classification time
- GPU build-kernel time
- CPU fallback time for singular buckets
- final near-field build time
- end-to-end solve time

Main question:
- does moving only the disjoint case to GPU materially reduce near-field setup time?

## Risks

### Risk 1: disjoint case is not dominant enough

Mitigation:
- measure case counts first before porting too much code

### Risk 2: singular cases dominate build time

Mitigation:
- stop after Stage 2 if the payoff is small

### Risk 3: device port of singular quadrature becomes numerically fragile

Mitigation:
- keep CPU singular fallback as the permanent safe mode until validated

## Recommended Next Implementation Step

Start with:

1. CPU near-pair classification and histogram
2. flattened pair metadata
3. GPU disjoint-only coefficient builder

Do not start with:

- full GPU `panelIA0()`
- merged all-case kernel
- direct-sum GPU experiments

## Bottom Line

This is worth trying, but only in a staged way.

Most realistic path:
- first: classify on CPU, port only the disjoint case
- later: decide whether singular-case GPU work is justified

That is the highest-value, lowest-risk way to attack the remaining `panelIA0()` bottleneck in the FMM near-field setup.
