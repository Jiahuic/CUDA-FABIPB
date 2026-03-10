## Near-field Roadmap

Date: March 10, 2026
Branch: `nearfield-roadmap`

### Scope

This roadmap covers the current GPU near-field implementation in the FMM path on `main`, identifies the likely bottlenecks, and defines the next experiments in the order they should be run.

This document is based on:

* current code in `src/fmm.c` and `src/gpu_backend_cuda.cu`
* existing comparison logs on the GPU machine

This document started from existing logs and was then updated after a destination-leaf grouped near-field implementation was benchmarked on the GPU machine.

Measured result on `test_proteins/1a63` averaged over 10 runs:

* mode 0 (interaction kernel):
  * `ttl = 5.794227 s`
  * `Near = 1.940889 s`
  * `build = 0.574423 s`
  * `kernel = 1.362885 s`
* mode 1 (destination-leaf grouped):
  * `ttl = 4.481293 s`
  * `Near = 0.585553 s`
  * `build = 0.561767 s`
  * `kernel = 0.020167 s`

Conclusion:

* destination-leaf grouped near-field is a solved milestone for the current codebase
* it should be the default GPU near-field path on `main`
* later grouping work should be compared against this baseline, not against the old interaction kernel

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

#### Bottleneck 1: grouped metadata setup

After destination-leaf grouping, setup/build is now much larger than the remaining near-field kernel.

From the averaged `1a63` grouped result:

* `build = 0.561767 s`
* `kernel = 0.020167 s`

Implication:

* the next near-field target is setup overhead, not repeated accumulation

#### Bottleneck 2: repeated near-field accumulation

From `build-cuda/compare_logs/20260309_154713` on `test_proteins/1a63`:

* CPU near-field total: `9.376543 s`
* GPU near-field total: `2.066644 s`

This was the main source of the GPU speedup in the interaction-kernel baseline, and destination-leaf grouping solved most of it.

Share of `applyFMM`:

* CPU: `9.376543 / 10.109454 = 92.8%`
* GPU: `2.066644 / 2.819454 = 73.3%`

Implication:

* on CPU, near-field dominates almost everything
* on GPU, near-field is still the largest remaining FMM stage

#### Bottleneck 3: host-device traffic every matvec

Current near-field apply still copies:

* `sgm` host -> device
* `pot` host -> device
* `pot` device -> host

for every matvec.

This is now timed explicitly, and on `1a63` it is small:

* `h2d = 0.001999 s`
* `d2h = 0.001611 s`

Implication:

* transfers are not the current dominant problem
* this should not be the next optimization target

#### Bottleneck 4: atomic contention

Current near-field kernel is interaction-centric:

* one thread per panel-pair interaction
* uses `atomicAdd` into destination outputs

This was the main issue in the old interaction kernel and is the reason destination-leaf grouping won so clearly.

Implication:

* the current kernel likely leaves performance on the table
* but not every grouping strategy is worth the metadata/setup overhead

#### Bottleneck 5: coefficient generation is still CPU-side

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

Status: completed

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

* now likely limited, because transfer time is very small on the tested case

Priority: low for the current near-field path

#### Experiment C: destination-leaf grouping

Goal:

* reduce contention without paying the setup cost of per-panel grouping

Why leaf grouping first:

* fewer groups than per-panel grouping
* less metadata overhead
* more likely to amortize setup cost

Measured payoff on `1a63`:

* near-field: `1.940889 s -> 0.585553 s` (`3.31x`)
* wall time: `5.794227 s -> 4.481293 s` (`1.29x`)

Status: completed and adopted

#### Experiment D: reduce grouped-metadata setup cost

Goal:

* reduce the one-time setup cost of the adopted destination-leaf grouped near-field path

Why now:

* setup is larger than the grouped kernel time on the tested case

Priority: highest follow-up

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

1. Reduce grouped-metadata setup cost.
2. Re-run averaged comparisons with `scripts/compare_nearfield_modes.sh`.
3. Only test finer grouping if it beats destination-leaf grouping end-to-end.
4. Only revisit coefficient generation after setup cost is better understood.

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

* reduce grouped near-field setup/build overhead without regressing the new destination-leaf kernel win
