## Current Status (March 21, 2026)

`main` now includes the validated production path for the PB solver:

- grouped GPU near-field apply
- cached-LU preconditioner
- GPU `M2L`
- GPU `L2P`
- GPU RHS setup
- CPU-default `Q2M` with GPU `Q2M` still available for experiments
- parallel preconditioner setup

Most importantly, the `panelIA0()` near-field setup bottleneck has now been addressed in a staged way:

- Stage 1:
  - exact `panelIA0()` case histogram added to near-field setup
  - on `1a63`, disjoint interactions dominate overwhelmingly:
    - `173,656,278 / 175,399,037`
- Stage 2:
  - disjoint-only near-field coefficient generation moved to GPU for `qOrder=1`
  - touching/singular cases remain on CPU
  - this is now merged into `main`

Validated `1a63` comparison:

- baseline GPU near-field build:
  - `ttl = 13.93 s`
  - `near-build = 7.34 s`
  - `coeff = 6.58 s`
- disjoint GPU near-field build:
  - `ttl = 8.36 s`
  - `near-build = 1.69 s`
  - `coeff = 0.89 s`
- correctness:
  - same `gmres-its = 13`
  - same solvation energy `-2385.222778`

This means the high-value part of the `panelIA0()` plan is complete:

- full all-case GPU `panelIA0()` is no longer required for the main paper result
- singular-case GPU port remains optional future work

Recommended near-field policy going forward:

- keep the disjoint-only GPU builder as the preferred `qOrder=1` setup path
- keep singular/touching cases on CPU unless a future branch shows clear additional payoff
- use [`scripts/compare_nearfield_build.sh`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/scripts/compare_nearfield_build.sh) for baseline vs disjoint-builder comparisons

Remaining paper-oriented priorities:

1. adaptive choice of `nLev` and `SepRat`
2. clean direct-sum appendix:
   - CPU direct
   - GPU direct
   - CPU FMM
   - hybrid GPU-FMM
3. benchmark tables and final figure selection

Historical planning notes follow below.

---

Below is a concrete, **paper-driven plan** to turn your existing **FMM-based Poisson–Boltzmann (PB) solver** into a **GPU-accelerated** method suitable for a strong computational math / scientific computing journal submission.

---

## Current Status (March 9, 2026)

Main should stay on the last stable fast baseline at commit `a243a0c`.

The slower experimental work has been preserved separately on branch:

* `nearfield-grouped-experiment`
* tip commit: `cbd69d0`

Summary of what was learned from that branch:

* flattened M2L caching reduced M2L time, but increased `setupFMM`
* destination-grouped near-field metadata and grouped kernel did not recover overall speedup on the tested machine
* the known-good GPU speedup still comes primarily from the original GPU near-field path

Solved near-field milestone:

* destination-leaf grouped near-field is now the preferred GPU near-field path
* averaged `1a63` comparison on the GPU machine showed:
  * wall time `5.794 s -> 4.481 s`
  * near-field stage `1.941 s -> 0.586 s`
  * near-field kernel `1.363 s -> 0.020 s`
* this is strong enough to keep as the default implementation and cite later in the paper as the current near-field acceleration result

Solved preconditioner milestone:

* cached local preconditioner block assembly is now validated as a real speedup path
* on `1a63`, keeping the same `gmres-its=13` and the same solvation energy:
  * `ttl` improved `89.68 s -> 53.51 s`
  * `psolve` improved `68.43 s -> 29.42 s`
  * preconditioner assembly improved `41.999 s -> 2.841 s`
* this is strong enough to keep and cite later as the current preconditioner acceleration result

Solved preconditioner LU milestone:

* cached LU reuse for the local preconditioner blocks is now validated on top of cached block assembly
* on `1a63`, keeping the same `gmres-its=13` and the same solvation energy:
  * `ttl` improved further `53.51 s -> 30.51 s`
  * `gmres` improved `44.83 s -> 18.67 s`
  * `psolve` improved `29.42 s -> 2.05 s`
  * preconditioner factor time dropped effectively `25.75 s -> 0`
* this is strong enough to keep and cite later as the current preconditioner LU reuse result

Mainline cleanup milestone before M2L:

* `main` now defaults to the validated fast path instead of the older development settings:
  * grouped near-field stays default (`-G=1`)
  * cached-LU preconditioner is the default (`-P=2`)
* `fabipb -h` now documents the runtime flags directly
* `-c=1` and `-C=1` remain available only as development compare modes and should not be used in benchmark tables
* next experimental work should start from this cleaned baseline, not from ad hoc flag combinations

Preconditioner development/debug note:

* `-C=1` is a development-only one-shot compare flag for `PtVfmm`
* it runs both the original and cached preconditioners once on the same vector and prints their difference
* it is not a benchmark mode and should be removed after preconditioner development is finished
* `-P=1` is the cached-block preconditioner mode
* `-P=2` is the cached-LU preconditioner mode and is now the preferred preconditioner path

See also:

* `docs/archive/nearfield_roadmap.md` for the archived detailed near-field bottleneck breakdown and experiment order

Implication for mainline work:

* make destination-leaf grouped near-field the default on `main`
* keep GPU M2L and later grouping variants as experimental branches until they show a clear end-to-end win

Near-field work that is still worth pursuing later:

1. keep `panelIA0()` cache construction as a one-time setup outside GMRES
2. reduce grouped-metadata setup cost
3. only revisit host-device transfer reductions if later kernels make copies matter
4. only revisit full GPU coefficient generation after the current near-field baseline is fully characterized

Preconditioner work that is still worth pursuing later:

1. consider batched CPU or GPU triangular solves only if `dgetrs` becomes worth targeting on the benchmark sizes of interest
2. remove `-C=1` once preconditioner development is considered complete

Main remaining runtime priorities after the current preconditioner work:

1. near-field cache build (`panelIA0()` setup) is now the largest remaining one-time cost on the large validated case
2. preconditioner solve (`dgetrs`) is smaller but still measurable
3. near-field kernel remains important, but no longer dominates the full run the way it did before grouping

Leaf-transform policy update:

* GPU `Q2M` was validated numerically and should remain available in the codebase
* current results show that `Q2M` is regime-dependent:
  * on moderate cases, CPU `Q2M` can still win because the current GPU path is dominated by copy and launch overhead
  * on deeper trees with very large leaf counts, GPU `Q2M` can become faster because the leaf-transform work is large enough to amortize the overhead
* the current runtime policy may stay conservative, but the final presentation goal should be an adaptive hybrid strategy rather than a fixed CPU-only or GPU-only rule
* final hybrid-mode goal:
  * choose CPU or GPU per stage based on problem size, depth, and leaf workload
  * in particular, allow `Q2M` to switch automatically when the leaf-transform workload crosses the GPU-beneficial threshold

Near-field setup update:

* thread-safe `panelIA0()` plus parallel near-field coefficient generation is now merged on `main`
* on `1a63`, this reduced:
  * near-field build `13.73 s -> 7.20 s`
  * coefficient generation `12.92 s -> 6.34 s`
  * total wall time `29.60 s -> 24.40 s`
* the next major algorithmic branch should therefore move to destination-grouped GPU `M2L`, not more near-field persistence work

How to treat destination-grouped near-field:

* destination-leaf grouping is now the adopted production path
* finer grouping variants should still stay on experiment branches until they beat the destination-leaf baseline

Grouped near-field research priorities:

1. reduce grouped-metadata setup cost
2. revisit destination-panel or finer grouping only if they beat destination-leaf grouping end-to-end
3. use the new near-field comparison script to report averaged comparisons over multiple runs

Required benchmark baseline to add:

* implement a direct GPU matvec baseline with no FMM
* this baseline should compute the full dense/direct interaction on GPU and use the GPU only as an accelerator for the direct matvec
* compare three paths in future benchmark tables:
  1. CPU baseline
  2. direct GPU matvec baseline without FMM
  3. GPU-accelerated FMM solver

Benchmark goal:

* show that GPU-FMM integration beats the non-FMM direct GPU baseline at relevant biomolecular problem sizes
* this comparison is important for the paper because it demonstrates that the FMM integration itself, not just “using a GPU,” is the source of the scalability gain

Next research branch after cleanup:

* start a dedicated `M2L` experiment branch from the cleaned `main`
* target: destination-grouped GPU cluster-cluster interaction (`M2L`)
* first implementation goal:
  1. flatten M2L interactions by destination cube
  2. compare GPU local expansions against CPU `transM2L`
  3. only then benchmark end-to-end solves

Direct GPU baseline implementation notes:

* first version should be treated as a benchmark/reference path, not the default solver
* v1 may precompute dense panel interaction coefficients with `panelIA0()` and upload them once to GPU
* this makes v1 suitable only for cases that fit GPU memory; large cases should fail clearly rather than silently distort the benchmark
* after v1 is working, the next step is to separate coefficient-build cost from steady-state dense GPU matvec cost

---

## 1) Pick a PB formulation that matches your current solver

Most FMM-PB solvers fall into one of these camps:

### A) Boundary integral PB (common for high accuracy)

* Reformulate PB (linearized or nonlinear) as coupled boundary integral equations on the molecular surface (and possibly an outer boundary).
* Discretize surface (triangles/panels) → dense interactions accelerated by FMM.
* GPU value: extremely strong (lots of kernel evaluations).

### B) Volume/particle or grid-based PB with FMM in a substep

* Less common; GPU value depends on what fraction is “FMM-heavy.”

**Plan assumes A (boundary integral),** because it’s the cleanest “FMM + GPU = big win” story. If yours is different, the GPU plan still applies, but the hotspots change.

---

## 2) Paper contribution (what you claim)

A publishable GPU paper usually needs **one main idea** plus rigorous benchmarking.

### Strong claim template

1. A **GPU-resident FMM** (or hybrid CPU/GPU) that accelerates PB electrostatics end-to-end.
2. A **numerically stable** implementation for screened and unscreened kernels relevant to PB.
3. Demonstrated **accuracy preservation** (same convergence/tolerance behavior) with **order-of-magnitude speedups** on realistic biomolecular systems.

If you can add *one* of these, it becomes more compelling:

* nonlinear PB (Newton/Picard) with GPU-accelerated matvecs
* multi-GPU scaling
* auto-tuned precision (mixed FP32/FP64 with certified error behavior)

---

## 3) Decide the GPU strategy: hybrid vs full GPU FMM

Given you already have a working FMM solver, the best thesis/paper path is usually: I would like to try Strategy 2

### Strategy 1: Hybrid FMM (fastest to publish)

* CPU: tree build + traversal schedule (or precomputed interaction lists)
* GPU: heavy kernels (P2P, P2M, M2M, M2L, L2L, L2P)
* You can still get big speedups because kernel evaluation dominates.

### Strategy 2: Fully GPU-resident FMM (higher ceiling)

* GPU: build tree (Morton code / LBVH), all operators on device
* More engineering; stronger “GPU paper” if you pull it off.

**Recommendation:** start with **hybrid**, then optionally extend.

---

## 4) Identify your kernels and map them to GPU work

For a classical FMM, your compute phases are:

* **P2P**: near-field direct interactions
* **P2M**: particles/sources → multipole coefficients at leaves
* **M2M**: upward pass aggregation
* **M2L**: interaction list translation (often the hotspot)
* **L2L**: downward pass
* **L2P**: evaluate local expansions at targets

### What usually dominates

* For many biomolecule-like distributions: **M2L + P2P** dominate.
* For highly clustered distributions: P2P can dominate.

### GPU mapping principles

* Structure-of-arrays memory
* Coalesced reads for source/target coordinates and coefficients
* Batch by level: process all boxes at same tree level together
* Use precomputed interaction lists for M2L (fixed-size: 189 neighbors in 3D uniform octree for Laplace-style; screened kernels may differ but still structured)

---

## 5) Kernel choices for PB electrostatics

Depending on PB variant:

### Linearized PB (screened Coulomb / Yukawa-type)

Kernel resembles:

* $$\frac{e^{-\kappa r}}{r}$$ and derivatives
  FMM for Yukawa is more complex than Laplace; your existing code likely handles this already.

### Nonlinear PB

Iterative scheme (Newton/Picard) repeatedly solves a linearized problem each iteration:

* GPU win is amplified because FMM matvec repeats many times.

**Paper angle:** “GPU-accelerated repeated FMM matvec for nonlinear PB.”

---

## 6) Implementation plan (milestones)

A realistic plan that turns into a paper:

### Milestone 0 — Profiling and ground truth (1–2 weeks equivalent effort)

* Add timers for each FMM stage: P2P, P2M, M2M, M2L, L2L, L2P, tree build, list build, solver iterations.
* Define accuracy targets:

  * potential error at sample points
  * surface charge density error (if BIE)
  * solvation energy error

Deliverable: a baseline performance/accuracy report.

---

### Milestone 1 — GPU P2P (near-field) + data layout (2–4 weeks)

* Port P2P to GPU first:

  * simplest to verify
  * often large speedup alone
* Build flattened neighbor lists on CPU and copy to GPU.

Deliverable: CPU tree + GPU P2P, verified against CPU.

---

### Milestone 2 — GPU M2L (main speed lever) (3–6 weeks)

* Port M2L translations:

  * batch per tree level
  * one thread block per target box or per interaction pair
* Store expansion coefficients efficiently:

  * for Cartesian expansions: coefficient arrays per box
  * for spherical harmonics: per-order coefficient arrays
* If memory is large: compress or use mixed precision for intermediate steps.

Deliverable: hybrid FMM where M2L+P2P are GPU, others CPU.

---

### Milestone 3 — Move the full pass to GPU (optional but strong) (4–8 weeks)

* Port P2M/M2M/L2L/L2P
* Keep tree/list generation on CPU initially
* Later: build tree on GPU if you want the “fully GPU FMM” claim.

Deliverable: end-to-end GPU compute path with minimal host-device traffic.

---

### Milestone 4 — Nonlinear PB acceleration (if applicable) (2–6 weeks)

* Integrate GPU-FMM into Newton/Picard iterations.
* Emphasize amortized gains: multiple matvecs per solve.

Deliverable: nonlinear PB benchmark with iteration counts and runtime.

---

## 7) Verification and numerical stability checklist

Reviewers will ask:

* Does GPU match CPU to within tolerance?
* How does error scale with expansion order / level / tolerance?
* Is the solver stable for high charge densities / large proteins?

**Minimum tests**

* analytic cases:

  * single sphere in electrolyte (known solutions)
  * two-sphere configuration (reference from high-accuracy BEM/direct)
* biomolecule cases:

  * small (1–5k atoms), medium (50k), large (200k+)
* sensitivity:

  * change tolerance; show monotone error/runtime tradeoff

---

## 8) Benchmarks and plots that make the paper

You want 4 “money figures”:

1. **Runtime breakdown** before/after GPU (stacked bars)
2. **Strong scaling vs N** (and optionally vs GPU count)
3. **Error vs runtime Pareto** (CPU direct, CPU FMM, GPU FMM)
4. **End-to-end PB solve time** for real biomolecules (with memory footprint)

Also report:

* achieved FLOP/s or effective throughput
* GPU occupancy / bandwidth limits (brief)

---

## 9) What makes it “journal-level innovative”

Pick one innovation hook:

### Hook A: “Level-by-level batched M2L with tensor-core-friendly structure”

If your expansions can be cast as small dense transforms, you can frame M2L as GEMM-like batches.

### Hook B: “Mixed precision with certified error control”

* FP32 for most operations, FP64 accumulation where needed.
* Provide a practical rule that preserves PB energy accuracy.

### Hook C: “Nonlinear PB with GPU-FMM in iterative solver”

* Emphasize repeated matvec acceleration and end-to-end speed.

### Hook D: “Multi-GPU distributed FMM for PB”

* Domain decomposition by Morton ranges; communicate multipoles across GPUs.
* Strong but more work.

---

## 10) Paper outline (maps to your implementation)

1. PB formulation and discretization (brief)
2. FMM algorithm used (the specific expansions and kernels)
3. GPU design

   * data layout
   * batching strategy
   * kernel details for P2P/M2L (and others as applicable)
4. Accuracy verification
5. Performance evaluation
6. Discussion and limitations

---

## 11) What I need from you (to lock this into a precise plan)

Paste these 6 items (short bullets are fine), and I’ll turn the above into a **step-by-step engineering checklist** with specific kernel designs:

1. PB type: linearized vs nonlinear
2. Formulation: boundary integral vs volume/grid
3. Kernel(s): Laplace $$1/r$$, Yukawa $$e^{-\kappa r}/r$$, plus derivatives?
4. Expansion type: spherical harmonics, Cartesian Taylor, or kernel-independent?
5. Current language/tooling: C++/CUDA? HIP? OpenMP? (and GPU model if known)
6. Where your profiler says time goes today (top 3 stages)

Once you share that, I’ll propose:

* exact GPU kernel mapping (thread/block)
* memory layout and coefficient storage
* and a publishable benchmark suite tailored to PB electrostatics.
