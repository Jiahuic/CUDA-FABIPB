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

Implication for mainline work:

* keep the stable near-field GPU baseline on `main`
* treat grouped near-field and GPU M2L as experimental branches until they show a clear end-to-end win

Near-field work that is still worth pursuing later:

1. keep `panelIA0()` cache construction as a one-time setup outside GMRES
2. reduce host-device traffic in repeated near-field applies
3. test coarser grouping strategies such as destination-leaf grouping before per-panel grouping
4. only revisit full GPU coefficient generation after the current near-field baseline is fully characterized

How to treat destination-grouped near-field:

* keep it as the innovation path, but not the default production path yet
* continue developing it on `nearfield-grouped-experiment`, not on `main`
* use `main` for stable correctness and timing baselines
* only merge it back when it shows a clear end-to-end improvement over the current atomic-kernel baseline

Grouped near-field research priorities:

1. reduce grouped-metadata setup cost
2. measure setup time, transfer time, and kernel time separately
3. test destination-leaf grouping before destination-panel grouping
4. revisit per-panel grouping only if the grouped apply phase is clearly superior when amortized over GMRES iterations

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
