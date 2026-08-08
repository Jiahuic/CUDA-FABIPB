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
FMM_R=1 FMM_DEPTH=8 FMM_GPU=1 FMM_PRECONDITIONER=3 GMRES_INITIAL=zero GMRES_RESTART=20 GMRES_MAX_ITER=100 GMRES_TOLERANCE=1e-4 FABIPB_RHS_TREE_THETA=0.3 FABIPB_RHS_THREADS=72 FABIPB_GPU_M2L_CHUNK_MIB=512 FABIPB_GPU_NEARFIELD_CHUNK_MIB=512 FABIPB_TIMEOUT=12h LIVE_LOG=1 OUT_DIR=results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_full scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

Generate and solve a fresh `sdens=2` case in a new directory:

```sh
FMM_R=0.5 FMM_DEPTH=9 FMM_GPU=1 FMM_PRECONDITIONER=3 GMRES_INITIAL=zero GMRES_RESTART=20 GMRES_MAX_ITER=100 GMRES_TOLERANCE=1e-4 FABIPB_RHS_TREE_THETA=0.3 FABIPB_RHS_THREADS=72 FABIPB_GPU_M2L_CHUNK_MIB=512 FABIPB_GPU_NEARFIELD_CHUNK_MIB=512 FABIPB_TIMEOUT=12h LIVE_LOG=1 OUT_DIR=results/fmm/6co8_fabipb_sdens2/zenodo_R05_depth9_full scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

`OUT_DIR` must be new or empty. `FABIPB_REUSE_MESH=1` trusts existing
`.face/.vert` artifacts and does not verify that they match the requested `R`.
Changing only `-R` while reusing an old prefix can therefore print a new mesh
request while solving the old mesh. Always confirm the raw vertex and face
counts in the log.

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

## Result Locations

Raw result directories are intentionally ignored by Git:

```text
results/fmm/6co8_fabipb_sdens1/zenodo_R1_depth8_full/
results/fmm/6co8_fabipb_sdens2/zenodo_R05_rhs/
results/debug/tabi_fmm_normal_component1/zenodo_sdens1_gpu_streaming/
```

The compact numerical record in this document is the version-controlled
milestone. Raw logs should be archived separately if they need long-term
retention.
