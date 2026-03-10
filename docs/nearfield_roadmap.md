## Near-field Roadmap

Date: March 9, 2026
Branch: `nearfield-roadmap`

### Scope

This roadmap covers the current GPU near-field implementation in the FMM path on `main`, identifies the likely bottlenecks, and defines the next experiments in the order they should be run.

This document is based on:

* current code in `src/fmm.c` and `src/gpu_backend_cuda.cu`
* existing comparison logs on the GPU machine

No new GPU experiments were run from this branch. Any payoff numbers below that are not directly visible in existing logs are marked as estimates.

### Current implementation

Current FMM/GPU split on `main`:

* `Q2M`, `M2M`, `M2L`, `L2L`, `L2P` stay on CPU
* only near-field direct interactions use the GPU

Current GPU near-field flow:

1. Build the flattened near leaf-pair list on CPU.
2. For every near panel pair, compute `panelIA0(pnlX, pnlY)` on CPU.
3. Store flattened arrays:
   * `src`
   * `dst`
   * `k0`
   * `k1`
   * `k2`
   * `k3`
4. Copy those arrays to GPU once.
5. For each GMRES matvec:
   * copy `sgm` to GPU
   * copy current `pot` to GPU
   * run one CUDA thread per cached panel-pair interaction
   * `atomicAdd` into destination panel outputs
   * copy `pot` back to CPU

So the current path is:

* CPU-built coefficient cache
* GPU-applied direct-sum accumulation

### Current bottlenecks

#### Bottleneck 1: repeated near-field accumulation

From `build-cuda/compare_logs/20260309_154713` on `test_proteins/1a63`:

* CPU near-field total: `9.376543 s`
* GPU near-field total: `2.066644 s`

This is already the main source of the GPU speedup, but it is still the largest FMM stage on GPU.

Share of `applyFMM`:

* CPU: `9.376543 / 10.109454 = 92.8%`
* GPU: `2.066644 / 2.819454 = 73.3%`

Implication:

* on CPU, near-field dominates almost everything
* on GPU, near-field is still the largest remaining FMM stage

#### Bottleneck 2: host-device traffic every matvec

Current near-field apply still copies:

* `sgm` host -> device
* `pot` host -> device
* `pot` device -> host

for every matvec.

This is visible in the code but is not yet timed separately on `main`.

Implication:

* even if the kernel math improves, repeated transfers can cap speedup
* this becomes more important as the GPU kernel gets faster

#### Bottleneck 3: atomic contention

Current near-field kernel is interaction-centric:

* one thread per panel-pair interaction
* uses `atomicAdd` into destination outputs

This is simple and robust, but it means many threads can contend for the same destination panel.

Implication:

* the current kernel likely leaves performance on the table
* but not every grouping strategy is worth the metadata/setup overhead

#### Bottleneck 4: coefficient generation is still CPU-side

Near-field coefficients are computed with `panelIA0()` on CPU.

This is amortized across GMRES iterations, so it is not the first target, but it still matters for:

* first-call latency
* setup-heavy runs
* any attempt to make the full path more GPU-resident

### Known failed direction

The per-destination-panel grouped near-field experiment did not beat the stable baseline end-to-end on the tested machine.

What was learned:

* expensive metadata rebuild can wipe out the kernel-side gains
* per-panel grouping is too fine-grained to adopt as the default next step

That experiment remains useful as a research direction, but it should not drive `main`.

### Payoff ceiling from current logs

These are upper bounds inferred from the current `1a63` GPU log, not measured improvements.

Current GPU run:

* `ttl time = 7.064021 s`
* `gmres = 5.757204 s`
* `applyFMM = 2.826326 s`
* `Near = 2.066644 s`

If near-field became free inside `applyFMM`, the best possible speedups would be approximately:

* `applyFMM`: `2.826 / (2.826 - 2.067) = 3.72x`
* `gmres`: `5.757 / (5.757 - 2.067) = 1.56x`
* end-to-end: `7.064 / (7.064 - 2.067) = 1.41x`

Implication:

* near-field is still the largest lever
* but the remaining end-to-end upside is no longer “another 2x” by itself

### Experiment roadmap

#### Experiment A: split near-field timing into setup, copies, and kernel

Goal:

* determine whether the next win is in transfers or in the CUDA kernel itself

Required measurements:

* one-time coefficient build time
* H2D copy time per matvec
* CUDA kernel time per matvec
* D2H copy time per matvec

Expected payoff:

* not a speedup by itself
* highest diagnostic value
* required before more kernel restructuring

Priority: highest

#### Experiment B: keep vectors resident longer across matvecs

Goal:

* reduce the repeated `sgm` / `pot` transfer cost

Minimal version:

* hold `sgm` and `pot` device buffers persistently
* avoid reallocating
* reduce copy traffic where possible

Stronger version:

* keep more of the FMM matvec path on GPU-adjacent data structures

Expected payoff:

* estimate: `5%` to `15%` end-to-end on current GPU path
* payoff grows if the kernel itself becomes faster

Priority: high

#### Experiment C: destination-leaf grouping

Goal:

* reduce contention without paying the setup cost of per-panel grouping

Why leaf grouping first:

* fewer groups than per-panel grouping
* less metadata overhead
* more likely to amortize setup cost

Expected payoff:

* estimate: `10%` to `25%` reduction in near-field apply time if grouping is effective
* end-to-end estimate: `3%` to `10%`

Priority: high, but only after Experiment A

#### Experiment D: revisit grouped kernels only after timing split

Goal:

* test grouped accumulation strategies with setup cost accounted for explicitly

Candidate variants:

* destination-leaf grouped kernel
* coarse destination-panel bins within leaf
* shared-memory block reductions with fewer final atomics

Expected payoff:

* unknown
* only worth continuing if steady-state kernel time drops enough to justify setup cost

Priority: medium

#### Experiment E: GPU coefficient generation for `panelIA0()`

Goal:

* move near-field coefficient generation off the CPU

Why later:

* higher engineering cost
* not the first-order repeated cost in current GMRES runs
* should only be attempted after setup/apply/transfer costs are separated clearly

Expected payoff:

* improves setup latency
* limited end-to-end impact unless setup is a large fraction of the target workload

Priority: medium-low

### Recommended execution order

1. Add timing split for near-field setup, H2D, kernel, D2H.
2. Run `1ajj` and `1a63` on the GPU machine and record the split.
3. If transfers are large, do persistent-device-buffer cleanup first.
4. If kernel dominates, test destination-leaf grouping before any finer grouping.
5. Only revisit coefficient generation after the above are measured.

### Success criteria

A near-field change should be considered good enough to keep only if it satisfies all of:

* no regression in solvation energy or GMRES iteration count
* no regression on small case wall time
* clear reduction in either:
  * near-field apply time
  * or total GMRES time
* setup overhead is accounted for explicitly in the comparison

### Immediate next coding task

The best next code task is:

* instrument the current near-field path into four parts:
  * coefficient build
  * H2D copies
  * kernel
  * D2H copy

This should be done before attempting another grouping kernel on top of `main`.
