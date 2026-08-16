# 6CO8 GPU FABIPB Milestone

Date: 2026-08-08

## Outcome

FABIPB completed the full 1,576,628-charge Zenodo 6CO8 calculation on
paper-scale NanoShaper surfaces for the first time. The fine `sdens=2` solve
converged to the requested GMRES tolerance, and the `sdens=1` solve reached a
residual within a factor of 1.51 of that tolerance before the 100-iteration
cap.

This milestone establishes that the full-virus FABIPB formulation can be
meshed, assembled, solved, and evaluated without truncating the charge set,
overflowing GMRES indexing, or exhausting GPU memory. Accuracy relative to
TABI-PB remains a separate follow-up task.

## Input And Reference

```text
input:       test_proteins/ZIKV_6CO8_zenodo.pqr
SHA-256:     b3d52572aa5076f66d84b49aeb8b631770b66041d49c242d6c04773a33b16e69
charges:     1,576,628
zero radius: 18,552 (retained as charges and included in NanoShaper XYZR)
eps1/eps2:   4 / 80
kappa:       0.125700
TABI-PB:     -117,561.982149 kcal/mol
paper:       -117,632.1 kcal/mol
```

The host used for the recorded runs has an Intel Xeon w9-3475X with 36 cores
and 72 logical CPUs. The GPU exposed 47.365 GiB total device memory in the run
logs. The sandbox cannot currently query the GPU model, so the model and driver
must be recorded from the GPU host before publication.

## Recorded Results

| Mesh | FABIPB `R` | Vertices | Panels | Area (A^2) | GMRES | Residual | Energy (kcal/mol) | Difference from TABI-PB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `sdens=1` | 1.0 | 5,109,760 | 10,222,292 | 3,227,710.506466 | 100, `info=1` | 1.509927e-4 | -123,418.693595 | 4.9818% more negative |
| `sdens=2` | 0.5 | 20,832,704 | 41,668,120 | 3,399,427.960540 | 37, `info=0` | 8.322437e-5 | -114,103.117477 | 2.9422% less negative |

The `sdens=1` energy is approximate because it did not satisfy the requested
`1e-4` tolerance. The `sdens=2` energy is a converged result for the recorded
FABIPB discretization and solver settings. The 7.55% change between the two
FABIPB energies shows that mesh convergence and the panel/nodepatch accuracy
gap still need study.

The independently generated `sdens=1` FABIPB mesh differs slightly from the
preserved TABI-PB mesh (5,109,746 vertices and 10,222,264 faces). Same-density
does not mean byte-identical mesh; exact operator comparisons must reuse one
set of `.face/.vert` files.

## Successful Configuration

Both full solves used:

```text
mesh backend:               NanoShaper
FMM order:                  adaptive (-1)
quadrature order:           1
dielectrics:                4 / 80
FMM depth:                  8 for sdens=1, 9 for sdens=2
preconditioner:             diagonal (P=3)
GMRES initial guess:        zero
GMRES restart:              20
GMRES maximum iterations:   100
GMRES tolerance:            1e-4
RHS:                        charge tree, theta=0.3, 72 threads
GPU M2L chunk:              512 MiB
GPU nearfield chunk:        512 MiB
```

The explicit `theta=0.3` RHS setting is an accuracy/runtime choice, not the
code default. Relative to `theta=0.2`, sampled full-virus RHS differences were
0.0136% for component 0 and 0.1288% for component 1. On `1a63`, the component
errors against direct RHS were 5.10e-4 and 2.39e-3 at `theta=0.3`.

That RHS ratio applies only to the RHS. The post-solve energy treecode has its
own `FABIPB_ENERGY_TREE_THETA` (default 0.2), deliberately separate: loosening
the RHS to 0.3 is a reasonable trade because setupRHS runs once per solve, but
applying the same 0.3 to the energy cost 70x accuracy on `1a63` (7.2e-6 ->
5.2e-4 against the theta->0 limit) for a stage that also runs only once.

## What Made The Full Solve Possible

1. **Zenodo PQR handling.** Zero-radius atoms are retained in the charge list
   instead of being dropped. They remain available to the electrostatic RHS
   and energy calculation while the meshing path handles their radius safely.
2. **Paper-scale NanoShaper meshes.** FABIPB maps `R` to NanoShaper
   `Grid_scale=1/R`, so `R=1` means `sdens=1` and `R=0.5` means `sdens=2`.
   Mesh artifacts can be preserved and reused for matched comparisons.
3. **Tree-accelerated RHS.** The direct panel-charge work is 16.1 trillion
   pairs at `sdens=1` and 65.7 trillion at `sdens=2`. A charge tree replaces
   the all-pairs CPU path, and independent per-thread derivative workspaces
   allow 72 pthread workers without shared scratch races.
4. **Large GMRES indexing.** GMRES workspace sizes and column offsets use
   `size_t`. This removes 32-bit multiplication overflow for the 83,336,240
   unknown `sdens=2` system. The program reports the workspace requirement
   before allocation; restart 20 required 14.902 GiB.
5. **GPU M2L streaming.** The `sdens=2` M2L operator contains 546,439,494
   pairs and 4,042,871,372 coefficients. A full device cache would require
   66.489 GiB and the coefficient offsets exceed signed 32-bit range. The GPU
   path keeps cube state resident and applies destination-aligned M2L chunks.
6. **GPU nearfield streaming.** A full nearfield cache would require about
   300 GiB for `sdens=2`. The streaming path moves bounded interaction chunks
   while panel geometry and vectors remain resident. Disjoint `qOrd=1`
   interactions run on the GPU; shared-vertex, shared-edge, and self-panel
   cases retain the specialized panel formulas.
7. **Failure diagnostics.** GPU cache estimates, available device memory,
   chunk counts, exact fallback reasons, GMRES residuals, and per-stage times
   are logged. This distinguished memory-capacity failures from numerical
   failures.
8. **Reproducible runner.** `scripts/run_6co8_fabipb_fast.sh` records the
   command, configuration, hashes, mesh/RHS/GMRES summary, and stage timings in
   one output directory.

The forced M2L streaming implementation was validated on `1a63` against the
CPU operator:

```text
applyFMM max absolute difference: 2.081668e-17
applyFMM relative L2 difference:  5.671110e-16
```

This is roundoff-level agreement. The full-virus runs additionally completed
all streamed M2L and nearfield chunks for every GMRES matvec.

## Commands

Generate and solve a fresh `sdens=1` case in a new directory:

```sh
SDENS=1 OUT_DIR=results/fmm/6co8_fabipb_sdens1/zenodo_sdens1_depth8_full scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

Generate and solve a fresh `sdens=2` case in a new directory:

```sh
SDENS=2 FMM_DEPTH=9 OUT_DIR=results/fmm/6co8_fabipb_sdens2/zenodo_sdens2_depth9_full scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

`OUT_DIR` must be new or empty. `FABIPB_REUSE_MESH=1` trusts existing
`.face/.vert` artifacts and does not verify that they match the requested `R`.
Changing only `-R` while reusing an old prefix can therefore print a new mesh
request while solving the old mesh. Always confirm the raw vertex and face
counts in the log.

The runner now exposes `SDENS`, not the internal FABIPB `R` parameter. It
translates `R=1/SDENS` for NanoShaper. This makes the command and result names
consistent with TABI-PB, but it does not claim that MSMS and NanoShaper have
identical density semantics: MSMS and NanoShaper use different resolution
controls and can produce different topology and panel counts at the same
nominal density. Exact comparisons should therefore reuse one saved mesh.

## Runtime Record

| Stage | `sdens=1` | `sdens=2` |
|---|---:|---:|
| Total | 4,429.31 s | 7,619.25 s |
| Load panels | 23.97 s | 28.69 s |
| Initialize kernels | 6.91 s | 29.28 s |
| Set up FMM | 6.10 s | 25.53 s |
| Set up RHS | 129.80 s | 684.16 s |
| GMRES | 2,994.99 s | 4,961.18 s |
| Energy treecode | 1,267.42 s | 1,889.97 s |

Nearfield remained the dominant GMRES cost. It averaged 23.15 seconds per
matvec at `sdens=1` and 88.27 seconds at `sdens=2`. The next performance work
should target nearfield throughput and the post-solve energy treecode, while
the next accuracy work should compare the panel-Galerkin and TABI nodepatch
representations on one byte-identical mesh.

## Optimized Rerun (2026-08-15 / 16)

The `sdens=1` case was rerun on the saved mesh with the settings the original
used (`-a=20`, `-t=8`, `-P=3`, `eps1=4`, `eps2=80`, tolerance `1e-4`,
`theta=0.3`), then rerun again after further work. Numbers below are the
latest measured values.

### Energies changed, deliberately

A defect was found in `transL2L` while parallelising the downward pass: it
filled its `fcnBuf` scratch to `ordOut` but indexed it to `ordIn`, and in
variable-order mode (the default) the coarser parent carries the higher order,
so `ordIn > ordOut` is the normal case. Entries in `(ordOut, ordIn]` were read
uninitialised, picking up whatever the previous call left in the shared
file-scope buffer. See the commit for the evidence that the fix is right
rather than merely different.

So energies moved by roughly 8e-5 relative. The pre-fix `sdens=1` value was
`-123,418.693595`; it is now `-123,425.081706` at `info=1`, 100 iterations,
residual `1.500076e-4`. The recorded-results table above is the original
milestone record and is left as it stands.

### sdens=1, measured end to end

| Stage | Recorded | Optimized | Speedup |
|---|---:|---:|---:|
| Total | 4,429.31 s | **870.95 s** | **5.09x** |
| Load panels | 23.97 s | 9.07 s | 2.64x |
| Set up RHS | 129.80 s | 127.62 s | 1.02x |
| GMRES | 2,994.99 s | 451.90 s | 6.63x |
| Energy treecode | 1,267.42 s | 268.71 s | 4.72x |

Inside GMRES, from the earlier instrumented run:

| | Recorded | Optimized | Speedup |
|---|---:|---:|---:|
| Nearfield | 2,430.67 s | 141.39 s | 17.19x |
| Nearfield GPU timer | 2,425.41 s | 135.70 s | 17.87x |
| Matvec total | 2,851.33 s | 559.00 s | 5.10x |

### sdens=1 after GPU setupRHS and scratch retuning

`results/fmm/6co8_fabipb_fast/sdens1_scratchtune`, total **700.83 s**.

**This run is not directly comparable to the 870.95 s above**, for two reasons
that have to be separated before quoting any ratio:

1. The 870.95 s run **did not converge** -- `info=1`, 100 iterations (the cap),
   residual `1.500076e-4`. This one converged: `info=0`, **87 iterations**,
   residual `9.865349e-05`. Comparing wall-clock totals therefore compares 87
   matvecs against 100. Per iteration GMRES went 4.519 s -> 4.268 s, and
   nothing in either change touches the matvec, so treat that 5% as run-to-run
   variance rather than a result.
2. Two changes landed between them: `setupRHS` moved to the GPU charge-tree
   kernel, and the charge-tree scratch was resized by thread count.

Isolating the stages that are independent of iteration count:

| Stage | CPU | GPU, 256 MiB budget | GPU, thread-sized | Net |
|---|---:|---:|---:|---:|
| Set up RHS | 127.62 s | 78.64 s | **58.72 s** | 2.17x |
| Energy treecode | -- | 268.71 s | **233.11 s** | 1.15x |

So the scratch retuning alone is worth 1.34x on `setupRHS` and 1.15x on the
energy stage here -- about 55 s, or 6-7% of the run. That is much less than the
2.17x / 1.56x it gives on 7A6A, and the reason is visible in the tuning data:
`sdens=1` runs at `derivMax=8`, where the thread-count curve is nearly flat,
whereas 7A6A's production config sits at `derivMax=6`, where the old fixed byte
budget overshot the optimum by 4x. The retuning matters most at low `derivMax`.

Note the previously reported 78.64 s `setupRHS` figure was measured at the
mistuned 46,976 threads, against a ~16k optimum, so it understated the GPU
kernel.

Energy `-123,425.979217`, against `-123,425.081706` for the 100-iteration run:
5.9e-5 relative, consistent with the two runs stopping at different residuals.

### sdens=1 with the Arnoldi orthogonalisation threaded

`results/fmm/6co8_fabipb_fast/sdens1_basisthreads`, total **652.99 s**.

| Stage | Serial basis | Threaded basis | Speedup |
|---|---:|---:|---:|
| GMRES `basis` | 58.97 s | **10.41 s** | **5.66x** |
| GMRES total | 371.34 s | 322.47 s | 1.15x |
| Run total | 700.83 s | 652.99 s | 1.07x |

Same 87 iterations, same final residual `9.865349e-05`, energy
`-123,425.979217` agreeing to every printed digit. That is not proof of bit
identity -- a reassociated sum shifts things around 1e-12 relative, below the
printed precision -- but it confirms the chunked reduction does not perturb the
Krylov trajectory.

This one was not an algorithmic problem. The ddot/daxpy pair already went
through OpenBLAS; it was pinned to a single thread by with_benchmark_env.sh,
which the runner never overrode, so 163 MB vectors were swept on one core. It
is the third time a runner-level override has hidden a binary default (after
the thread pools in ec7a1a4 and the `-Q=0` hardcode).

### Energy kernel: measured footprint sensitivity

Nsight Compute is unavailable on this machine -- `ERR_NVGPUCTRPERM`, with
`RmProfilingAdminOnly: 1`, so performance counters need root. The same question
was settled with a timing-only probe instead: the kernel was patched so threads
share derivative slabs, holding thread count, arithmetic, and coalescing fixed
while the footprint shrinks. Results are garbage under the races; only the time
is meaningful. On 7A6A at `derivMax=9`, 22,784 threads:

| Footprint | Energy stage | vs. current |
|---:|---:|---:|
| 248.6 MiB (as shipped) | 37.17 s | -- |
| 124.3 MiB | 21.54 s | 1.73x |
| 44.7 MiB | 16.07 s | 2.31x |
| 11.2 MiB | 15.43 s | 2.41x |
| 2.8 MiB | 14.92 s | 2.49x |

So the kernel is footprint-bound, and roughly 2.5x is the most any footprint
reduction can return. About 70% of the slab is intermediate levels the
contraction never reads (495 of 715 doubles per array at `derivMax=8`, 715 of
1001 at 9); they exist only because the recurrence builds level p from level
p+1. The dependency is narrow -- row `iRow` of level p needs only rows `iRow-1`
and `iRow-2` of level p+1, and the loop already runs `iRow` outermost -- so
keeping two rows per level and fusing the level-0 contraction into the row loop
would cut the footprint 3.09x at `derivMax=9`, to about 80 MiB. Interpolating
the table, that is worth roughly 1.9-2.0x on the energy stage. Not yet
implemented.

### Rolling derivative scratch in the energy kernel

The pyramid materialised every level when the contraction only reads level 0;
two rows per level suffice, with the level-0 contraction fused into the row
loop. Slab per array goes from sum_m nMom[m] to 2*nMom[derivMax]. The smaller
slab also invalidated the flat per-SM thread count: measured optima are
(slab 224 -> 32,768), (480 -> 22,784), (880 -> 16,384), where threads*sqrt(slab)
is constant to 3%, so that is the rule now used.

Energy stage, before -> after:

| Case | derivMax | Slab | Before | After | Speedup |
|---|---:|---:|---:|---:|---:|
| 7A6A `-p=4` | 5 | 252 -> 224 | 2.921 s | 2.518 s | 1.16x |
| 7A6A `-p=6` | 7 | 660 -> 480 | 14.816 s | 8.401 s | 1.76x |
| 7A6A `-p=8` | 9 | 1430 -> 880 | 37.227 s | 26.275 s | 1.42x |
| capsid `sdens=2` | 9 | 1430 -> 880 | 1,548.73 s | 1,223.75 s | 1.27x |
| capsid `sdens=1` | 8 | 990 -> 660 | 233.10 s | 226.13 s | 1.03x |

**The gain is highly configuration-dependent and falls off on the real capsid.**
It is largest where the cluster expansions dominate the walk; the leaf-charge
direct summation is untouched by the scratch change, and where it takes a bigger
share -- as at `sdens=1` -- there is almost nothing to win. Forward projections
from the footprint probe overestimated this stage twice (1.9-2.0x predicted,
then 1.42x measured on 7A6A, then 1.27x and 1.03x on the capsid); the probe
varies allocation and touched set together, while the rolling window shrinks
them by different amounts.

Correctness held at every scale. Energies match the pyramid version to every
printed digit at all three orders on 7A6A and at every thread count swept, and
on the capsid both densities reproduced their previous energies exactly --
`-114,082.750614` at `sdens=2` over 41,668,120 panels and `-123,425.979217` at
`sdens=1`, with identical iteration counts and residuals. Arithmetic and its
order are unchanged; only the storage location moved.

### sdens=2

| Stage | Recorded | Optimized | Speedup |
|---|---:|---:|---:|
| Total | 7,619.25 s | **2,524.93 s** | **3.02x** |
| GMRES | 4,961.18 s | 852.75 s | 5.82x |
| Nearfield | 3,354.73 s | 266.66 s | 12.58x |
| M2M | 74.63 s | 3.52 s | 21.20x |
| L2L | 57.91 s | 3.27 s | 17.71x |
| Energy treecode | 1,890 s | 1,223.75 s | 1.54x |
| Set up RHS | 684.16 s | 324.06 s | 2.11x |
| M2L | 941.25 s | 445.93 s | 2.11x |
| GMRES basis | -- | 18.33 s | 5.39x vs serial |

The M2L figure validated the 2.13x projected from a forced-streaming 7A6A case,
and Q2M went 98.5 s -> 30.3 s (3.24x) once the runner stopped overriding the
GPU Q2M default. 37 iterations, residual `8.324220e-05`.

`results/fmm/6co8_fabipb_sdens2/scratchtune_rhsgpu`, 37 iterations, residual
`9.355239e-05`, energy `-114,082.750614`.

The GPU `setupRHS` kernel and the thread-sized scratch both landed after the
previous run at this density, and they split exactly as the tuning sweep
predicted: `setupRHS` 689.61 -> 320.55 s (2.15x), while the energy stage was
flat at 1,542.71 -> 1,548.73 s. `derivMax=9` is where the thread-count curve
levels off, so there was nothing there for the retuning to win; the RHS kernel
had been sitting at 46,976 threads against a ~16k optimum, and that is where
the gain came from. Earlier intermediate totals at this density of 3,846 s and
3,225.61 s are superseded.

`basis` was 98.75 s in this run, which still used the pre-threading binary; see
the sdens=1 section for what threading it is worth.

Energy `-114,101.694486` against the recorded `-114,103.117477`, a 1.25e-5
relative shift consistent with the `transL2L` fix.

### Reading compare-mode timings

`FABIPB_ENERGY_MODE=compare` runs **both** evaluators, so its energy stage is
the sum of the two and must not be compared against a recorded single-evaluator
figure. Doing that is what made the `sdens=2` energy stage look like a 2x
regression when the correct split was charge-tree 1,885 s (matching the
recorded 1,890 s) and panel-tree 1,594 s, i.e. an improvement. Both evaluators
at `sdens=1`:

```text
charge-tree  -59.151987570342845   1279.22 s
panel-tree   -59.154442436439652    445.28 s   (rel-diff 4.15e-5)
```

The panel-tree value is the more accurate of the two; see the theta->0
convergence check in the source comment on `energyTreeTheta()`. The GPU
evaluator reproduces the CPU one exactly and is 1.66x faster at `sdens=1`
(268.7 s against 445.3 s), measured by holding the solution fixed and toggling
only the evaluator with `FABIPB_ENERGY_GPU=0`. Comparing `-g=1` against `-g=0`
instead compares two different GMRES solutions and shows a spurious 4e-5
difference.

### New machinery, at scale, all within budget

```text
sdens=1  special cache   138,366,798 pairs    5.155 GiB host   168 chunks
         leaf-pair table  25,529,703 pairs    0.476 GiB device
sdens=2  special cache   561,290,776 pairs   20.910 GiB host   582 chunks
         leaf-pair table  99,879,047 pairs    1.860 GiB device
         pinned staging   6/6 buffers         0.500 GiB host
```

### Where the time now sits

At `sdens=1` the run is 871 s and fairly flat. At `sdens=2` the balance is
different because M2L cannot fit the device cache (66.5 GiB needed against
46.8 GiB free) and therefore streams, rebuilding its coefficients on every
matvec: energy 1,594 s, M2L 950 s, setupRHS 685 s. The M2L coefficient build
has since been threaded and `setupRHS` has a device implementation; neither is
reflected in the table above.

Note also that phases alternate between host and device, so `nvidia-smi` shows
the GPU idle for long stretches -- `setupRHS` alone was 685 s at 0% GPU before
it was ported.

## Result Locations

Raw result directories are intentionally ignored by Git:

```text
results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_full/
results/fmm/6co8_fabipb_sdens2/zenodo_R05_rhs/
results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_optimized/
results/debug/tabi_fmm_normal_component1/zenodo_sdens1_gpu_streaming/
```

The optimized rerun reused the saved mesh from the original `sdens=1`
directory by hard-linking `input.face`, `input.vert` and `input.xyzr` into the
new output directory, so NanoShaper never ran and both solves are on a
byte-identical surface:

```sh
SRC=results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_full
OUT=results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_optimized
mkdir -p "$OUT"
for f in input.face input.vert input.xyzr; do ln "$SRC/$f" "$OUT/$f"; done

ALLOW_EXISTING_OUT_DIR=1 \
FMM_R=1 FMM_DEPTH=8 FMM_GPU=1 FMM_PRECONDITIONER=3 \
GMRES_INITIAL=zero GMRES_RESTART=20 GMRES_MAX_ITER=100 GMRES_TOLERANCE=1e-4 \
FABIPB_RHS_TREE_THETA=0.3 FABIPB_ENERGY_MODE=compare \
FABIPB_TIMEOUT=12h LIVE_LOG=1 OUT_DIR="$OUT" \
scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

The compact numerical record in this document is the version-controlled
milestone. Raw logs should be archived separately if they need long-term
retention.
