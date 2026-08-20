# TABI-PB/FMM Large-Protein Diagnosis

Date: 2026-08-01

## Question

Small proteins produce plausible FMM energies, while the full 6CO8 virus does
not. Is there a solver failure caused directly by the number of atoms?

## Current Answer

The matched tests do **not** show an abrupt atom-count failure through 93,744
atoms and 613,988 panels. An independent same-setting 1AON check also converged
for a 120,120-atom, roughly 885k-panel protein assembly. FMM is consistently a
few percent more negative than TABI-PB on fine matched NanoShaper surfaces
where both methods have been compared. The full-virus order-of-magnitude
discrepancy is instead correlated with severe surface coarsening and the
different behavior of panel-Galerkin and nodepatch discretizations on that
coarse surface.

## Matched Protocol

For every matched-mesh case below, unless explicitly noted:

- TABI-PB and FMM read the same PQR.
- TABI-PB generates the NanoShaper `sdens=1` surface.
- `TABIPB_KEEP_MESH=1` preserves that surface.
- FMM reuses the exact same `.face/.vert` files.
- Dielectrics are 4/80, salt is 0.15 M, and the tolerance is `1e-4`.
- Neither method uses a preconditioner.
- FMM uses at most 100 iterations.

TABI-PB has one unknown pair per mesh vertex (nodepatch). FMM has one unknown
pair per triangular face (piecewise-constant Galerkin), so the common geometry
does not make the two discrete linear systems identical.

## Size-Sweep Results

| Case | Atoms | Vertices | Faces/FMM panels | TABI energy | FMM energy | FMM error vs TABI | TABI its/residual | FMM its/residual |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1a63 | 2,065 | 10,350 | 20,688 | -613.927196 | -636.118867 | -3.615% | 10 / 5.72e-5 | 77 / 9.63e-5 |
| 2h8h | 7,084 | 29,028 | 58,060 | -1,313.875138 | -1,358.914297 | -3.428% | 10 / 9.90e-5 | 82 / 9.77e-5 |
| Zika chain A | 7,534 | 31,490 | 62,972 | -788.184525 | -831.265006 | -5.466% | 10 / 4.58e-5 | 98 / 9.88e-5 |
| Zika asymmetric unit 1 | 26,277 | 106,640 | 213,264 | -2,447.695901 | -2,597.740938 | -6.130% | 10 / 8.79e-5 | 100 / 1.38e-4 |

The one-unit FMM result is capped slightly above tolerance, but it is stable:
GMRES(30) gave `-2597.661170` at residual `1.89e-4`, while Arnoldi-100 gave
`-2597.740938` at residual `1.38e-4`. The energy changed by only 0.0031%.

## Larger Fine-Mesh Extension

A two-unit Zika case extends the matched `sdens=1` comparison to 52,554 atoms,
205,396 vertices, and 410,796 triangular panels. FMM used depth 8 and the
block-LU preconditioner so that the larger solve reached the common tolerance:

| Case | Atoms | Vertices | Faces/FMM panels | TABI energy | FMM energy | Gap | TABI its/residual | FMM its/residual |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Zika asymmetric units 1-2 | 52,554 | 205,396 | 410,796 | -4,694.775671 | -4,942.184664 | 5.270% | 12 / 9.14e-5 | 30 / 8.36e-5 |

The surface has 36 connected components and area 129,565.733 A^2. TABI-PB and
FMM report the same area, and SHA-256 files recorded before and after FMM are
identical. FMM completed in 243.48 seconds, including 116.71 seconds for the
tree-accelerated RHS and 114.70 seconds for GMRES. This larger case remains in
the accepted 3-6% band and provides no evidence of an atom-count-driven solver
failure.

## Independent Large-Protein Convergence Check

RCSB `1AON` (GroEL/GroES/(ADP)7 chaperonin complex) was added as an independent
large protein assembly between the 52k-atom Zika two-unit case and the full
1.58M-atom virus. The official PDB was converted with PDB2PQR 2.1.1 using
CHARMM. Because the direct PDB2PQR output includes fixed-width records that the
current whitespace-token PQR parser can reject, this check used a compact
protein-only PQR preserving the fixed-column coordinates and final CHARMM
charge/radius fields.

TABI-PB was then run on the same compact PQR with `sdens=1`, matching the FMM
`R=1` NanoShaper grid scale. TABI-PB generated a nearly identical but not
byte-identical mesh: 442,616 vertices and 885,452 triangles, versus FMM's
442,612 vertices and 885,444 panels. The surface area differed by only
0.359717 A^2, or 0.000128%.

| Case | Atoms | Method | Vertices | Faces/panels | Energy | its/residual | Total time |
|---|---:|---|---:|---:|---:|---|---:|
| 1AON protein assembly | 120,120 | TABI-PB | 442,616 | 885,452 | -62,318.447134 | 8 / 8.69e-5 | 350.14 s |
| 1AON protein assembly | 120,120 | FMM | 442,612 | 885,444 | -63,022.184149 | 23 / 9.04e-5 | 951.55 s |

This result supports the view that the full 6CO8 problem is not simply "too
many atoms." A separate 120k-atom, 885k-panel protein converges under the same
100-iteration cap, while the full-virus `R=2` case fails at 100 iterations with
a residual of `4.86e-1`. The 1AON FMM energy is 1.129260% more negative than
TABI-PB, well inside the earlier 3-6% accepted band.

## Additional Large PDB Same-Mesh Batch

Three additional RCSB protein assemblies were tested with the exact
TABI-generated `sdens=1` mesh reused by FMM:

| Case | Atoms | Vertices | Faces/FMM panels | TABI energy | FMM energy | Gap | TABI its/residual | FMM its/residual |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 7A6A apoferritin | 66,456 | 210,270 | 420,580 | -46,523.936214 | -47,582.111392 | 2.274% | 9 / 7.53e-5 | 20 / 9.73e-5 |
| 3C92 20S proteasome | 93,744 | 306,916 | 613,988 | -14,977.434615 | -15,553.430853 | 3.846% | 15 / 6.95e-5 | 60 / 9.75e-5 |
| 6CVM beta-galactosidase | 63,980 | 194,514 | 388,960 | -24,910.119033 | -25,354.716494 | 1.785% | 9 / 5.36e-5 | 28 / 9.91e-5 |

All three FMM solves used depth 8 and block-LU preconditioning, reached the
requested `1e-4` tolerance, and passed SHA-256 mesh checks before and after FMM.
This extends the exact-mesh ladder past the two-unit Zika case and keeps the
FMM/TABI gap in the same few-percent regime.

## Self-Panel Result

Excluding identical source/target panels does not improve the fine-mesh
comparison:

| Case | FMM normal | FMM skip-self | Effect on agreement with TABI |
|---|---:|---:|---|
| 1a63 | -636.118867 | -637.881509 | worse |
| 2h8h | -1,358.914297 | -1,363.349885 | worse |
| Zika chain A, depth 6 | -831.265006 | -834.644944 | worse |

The dramatic skip-self change on coarse full-virus meshes is therefore not a
general correction. It is a coarse-discretization cancellation.

## Why 6CO8 Is Different

The paper/TABI-PB calculation uses NanoShaper `sdens=1` and 5,109,746
nodepatch elements, giving `-117,561.982149 kcal/mol` in the reproduced run.

The practical FMM virus diagnostics instead use:

- `R=8`, which maps to NanoShaper `Grid_scale=1/R=0.125`;
- only 108,528 triangular panels;
- an energy near `-1.266e6 kcal/mol` for the self-excluded formulation.

Thus `R=8` is **coarser**, not finer, than `R=1`. It is eight times coarser in
NanoShaper grid scale. The paper surface would also have roughly twice as many
faces as its 5.1 million vertices, so the practical FMM mesh is about two
orders of magnitude smaller in surface degrees of freedom.

An independent matched coarse R=10 experiment gave TABI-PB energy
`-1,450,973.574187 kcal/mol`, far from the paper. This proves that TABI-PB also
loses the paper result when forced onto the severely coarse surface. FMM and
TABI then disagree with each other because their discrete formulations react
differently to under-resolution.

## Code Findings

1. `src/fabipb.c` maps FMM `R` to NanoShaper grid scale as `1/R`. The naming is
   easy to misread when moving between TABI `sdens` and FMM `R`.
2. `src/numQuad.c:613` implements piecewise-constant Galerkin panel integrals,
   including separate disjoint, shared-vertex, shared-edge, and identical-panel
   formulas. TABI-PB is a vertex nodepatch method, so raw matrix entries and
   unknown vectors are not directly interchangeable.
3. The FMM energy scale and `L1/L2` formula match TABI-PB. On R=8, 99.77% of
   the erroneous energy comes from `L1 * potential[0]`, not unit conversion.
4. The accelerated RHS agrees with the direct RHS on the small validation case;
   it remains a candidate for a sampled large-case check, but there is no
   evidence that it creates a 10x energy error.
5. Tree depth must follow geometry, not atom count. For the 213k-panel Zika
   unit, depth 6 generated 371,372,582 near panel pairs, depth 7 generated
   83,062,312, and depth 8 generated 21,415,570. Depth 8 reduced the maximum
   leaf occupancy to 102 and completed the solve in about 3 minutes of GMRES.
6. The partial-restart GMRES exit bug was fixed by updating with only the
   Krylov columns computed in the final partial cycle. Current size-sweep
   results are post-fix.
7. The current PQR reader is fragile for fixed-width PDB2PQR output when fields
   touch, for example positive-to-negative coordinate transitions or overflowing
   atom serials. The 1AON convergence run used a compact protein-only PQR to
   avoid that parser issue; the reader should be hardened before direct PDB2PQR
   output is treated as a reliable large-input path.

## Resolution Sweep Result

The matched-mesh resolution sweep on the 26,277-atom Zika unit is complete:

| sdens | Faces | Components | Area | TABI energy | FMM energy | Gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 213,264 | 18 | 67,630.745 | -2,447.696 | -2,597.863 | 6.14% |
| 0.5 | 50,684 | 17 | 61,128.826 | -3,388.336 | -3,798.343 | 12.10% |
| 0.25 | 11,400 | 7 | 52,527.247 | -377,271.747 | -113.452 | 99.97% |
| 0.125 | 2,364 | 5 | 42,035.402 | -10,878.392 | +10,935.929 | 200.53% |

All final FMM rows use block-LU preconditioning and satisfy the `1e-4`
residual. TABI-PB also satisfies the same tolerance. The catastrophic coarse
results are therefore converged solutions of invalid under-resolved discrete
surfaces, not unfinished linear solves.

The surface topology changes between `sdens=0.5` and `0.25`: connected
components drop from 17 to 7 while surface area falls by another 14%. At
`sdens=0.125` only 5 components and 62.15% of the `sdens=1` area remain.

This directly explains the full-virus observation. FMM `R=8` maps to
`sdens=0.125`, the most under-resolved point tested. Reproducing the paper
requires moving toward `sdens=1`, not increasing `R`.

Fresh endpoint reruns reproduced the `sdens=1` and `sdens=0.125` energies
exactly to six decimal places. SHA-256 checks before and after FMM verified that
the TABI-generated `.face/.vert` files were not changed, so accidental
remeshing is ruled out.

## Full-Virus R=2 Attempt

The full 1,576,628-atom Zenodo PQR was also tested with FMM `R=2`, which maps
to NanoShaper grid scale `0.5`. The generated surface had 1,211,200 vertices,
2,425,520 panels, and area 2,910,484.965235 A^2. This is substantially finer
than the earlier coarse full-virus meshes.

The depth-8, cached-LU solve did not converge within the required 100-iteration
cap: its final residual was `4.863928e-01` (`info=1`). The program printed
`-1,618,041.875893 kcal/mol`, but that value is not a valid energy because the
linear solve is unconverged. Total runtime was 6,402.71 seconds, including
2,707.50 seconds in `setupRHS` and 2,446.35 seconds in GMRES.

This run does not test whether refinement resolves the physical energy gap;
it instead exposes a full-capsid conditioning/preconditioner problem. Earlier
coarse full-virus diagnostics found the same cached-LU failure mode while the
unpreconditioned path converged, so preconditioner choice must be isolated on
the finer mesh before interpreting its energy.

## Full-Virus R=1 Fine Mesh

The full Zenodo PQR was then tested with FMM `R=1` in mesh-only mode. Since
FMM maps `R` to NanoShaper `Grid_scale=1/R`, this is the fine `sdens=1`
paper-scale surface:

```text
atoms/charges:       1,576,628
zero-radius atoms:   18,552
mesh vertices:       5,109,760
mesh panels:         10,222,292
surface area:        3,227,710.506466 A^2
```

This confirms that the fine full-virus mesh generation itself is feasible and
that `R=1` reaches the expected paper-scale vertex count. The preserved mesh is
stored under `results/fmm/ZIKV_6CO8_zenodo_R1_mesh/`.

A setupFMM-only diagnostic on the preserved mesh also completed:

```text
leaf cubes:          1,242,319
near leaf pairs:     25,529,703
M2L pairs:           169,047,818
FMM cubes:           1,634,514
near panel pairs:    2,242,813,940
setupFMM time:       6.257298 s
```

A GPU-request setup diagnostic also completed on the same preserved mesh with
`-g=1`, but it intentionally stopped before matvec evaluation. Before host GPU
device visibility was fixed, a separate one-iteration `1a63` run reached the
nearfield backend and showed that FABIPB could not execute CUDA kernels from the
sandboxed environment:

```text
CUDA backend unavailable: no CUDA-capable device is detected
GPU backend requested but unavailable; using CPU nearfield path.
```

After host GPU device visibility was fixed and FABIPB was run outside the
sandbox, the CUDA path was exercised successfully. A converged `1a63` CPU/GPU
comparison with the same solver settings gave:

```text
CPU: info=0 iterations=20 residual=7.048789e-05 energy=-635.919931 time=6.478839 s
GPU: info=0 iterations=20 residual=6.649908e-05 energy=-635.922470 time=1.077429 s
```

The GPU log reports active GPU RHS, M2L, leaf, and nearfield caches. The
nearfield stage averaged 9.373 ms per call on GPU versus 241.216 ms per call on
CPU for this small case.

The first large same-mesh GPU validation used the 26,277-atom Zika unit-one
`sdens=1` mesh: 106,640 vertices and 213,264 panels. With the old RHS policy,
GPU matvec worked but RHS still used the CPU charge tree because the pair count
was slightly above the 5B CPU direct cap:

```text
CPU baseline:   RHS=47.704001 s GMRES=51.656658 s total=103.756452 s energy=-2597.863326
GPU tree RHS:   RHS=48.270219 s GMRES=5.664192 s  total=59.474114 s  energy=-2597.869357
```

The RHS policy was then changed to use a separate GPU direct-RHS cap:
`FABIPB_MAX_GPU_DIRECT_RHS_PAIRS`, default `200000000000`. This keeps CPU runs
on the conservative 5B cap while allowing GPU runs to use direct RHS for
medium-large cases. The same Zika unit-one GPU run now selects direct GPU RHS by
default:

```text
setupRHS direct pairs: panels=213264 charges=26277 pairs=5603938128
  limit=200000000000 mode=direct
GPU RHS cache: panels=213264 charges=26277 qOrder=1
GPU direct RHS: RHS=0.356403 s GMRES=5.560091 s total=11.202066 s energy=-2597.872667
```

This gives a 9.26x total speedup over the CPU baseline on the same mesh, while
the energy differs by only 0.009341 kcal/mol.

The same GPU policy was then checked on the larger two-unit Zika case with the
preserved `sdens=1` mesh: 52,554 charges, 205,396 vertices, and 410,796 panels.
This case has 21,588,972,984 panel-charge RHS interactions, still below the
200B GPU cap:

```text
CPU baseline:   RHS=116.710355 s GMRES=114.703824 s total=243.475790 s energy=-4942.184664
GPU direct RHS: RHS=1.316061 s   GMRES=12.188068 s  total=26.820925 s  energy=-4942.191462
```

The GPU two-unit run converged in 31 iterations with residual `8.731072e-05`.
The energy differs from the CPU FMM baseline by 0.006798 kcal/mol, confirming
that the GPU RHS/matvec path is numerically consistent for this larger matched
surface.

A full-virus coarse `R=8` smoke run was also completed on the full
1,576,628-charge Zenodo PQR. This mesh has only 53,072 vertices and 108,528
panels, so it is not the paper-scale calculation, but it exercises the full
charge list with direct GPU RHS:

```text
setupRHS direct pairs: panels=108528 charges=1576628 pairs=171108283584
  limit=200000000000 mode=direct
GPU RHS cache: panels=108528 charges=1576628 qOrder=1
```

The full-virus `R=8`, `P=2` solve did not converge within 100 iterations
(`final-residual=6.455501e-01`). The normal run then spent 362.872799 seconds in
the CPU post-solve treecode energy stage and printed a nonphysical positive
energy, so that energy is invalid. A new benchmark switch,
`FABIPB_STOP_AFTER_GMRES=1`, was added to skip that post-solve energy work during
large failed-solve diagnostics:

```text
FABIPB_STOP_AFTER_GMRES set: skipping post-GMRES treecode energy.
ttl time: 26.397655, gmres-its=100
Top-level stage times (s): loadPanel=2.408507 gkInit=0.156584 setupFMM=0.120496
  setupPC=0.635496 setupRHS=9.634088 gmres=13.441859 treecode=0.000000
```

This shows that direct GPU RHS and GPU FMM matvec can run against the full virus
on a small mesh. It does not solve the full-virus numerical problem: coarse
full-virus convergence/preconditioning is still poor, and final energy
evaluation remains CPU-bound when it is requested.

The full-virus `R=1` staged GPU run was also retried on the preserved mesh. With
`-P=2`, the run was stopped because cached-LU preconditioner setup begins before
RHS and consumed about 10 GB on CPU before any GPU work. With `-P=-1`, the run
reached mesh reuse and setupFMM, but RHS selected the CPU tree path:

```text
setupRHS: 16116751791376 panel-charge interactions (panels=10222292 charges=1576628)
  exceeds FABIPB_MAX_DIRECT_RHS_PAIRS=5000000000; using tree-accelerated RHS.
setupRHS direct pairs: panels=10222292 charges=1576628 pairs=16116751791376
  limit=5000000000 mode=tree theta=0.2
```

That no-preconditioner full-virus run was intentionally stopped during CPU tree
RHS. It had not reached GPU matvec evaluation.

No full-virus `R=1` energy has been computed yet. Even with the new GPU direct
RHS cap, the full-virus `R=1` pair count is `1.6116751791376e13`, still far
above the default 200B GPU cap. The next safe stage is either an explicit
override benchmark for full-virus GPU direct RHS or a more scalable batched/tree
GPU RHS path; a full solve should only follow after RHS timing and
preconditioner behavior are isolated.

## Next Work

1. Treat `sdens>=1` (`R<=1`) as the physical reference regime.
2. Put TABI-PB and FABIPB vectors in one discrete space before interpreting
   vector norms. Build an area-weighted triangle-to-vertex nodepatch mapping
   (and its vertex-to-triangle counterpart), then compare `b`, `v1`, `Av1`,
   `Minv_Av1`, and `x_after_iter1` point by point. The current fine 6CO8 run
   has 5,109,746 TABI vertex unknowns versus 10,222,264 FABIPB triangle
   unknowns, so its raw norm-only comparison is not an operator equality test.
3. After applying that mapping, compare `RHS[0]`, `A00`, and `A01` on the fine
   matched mesh. In the latest run, FABIPB's TABI-area-normalized RHS sum ratios
   are 0.999904 for component 0 and 1.067393 for component 1; the first residuals
   remain 0.222528 (FABIPB) and 0.072083 (TABI-PB).
4. Replace the full-virus CPU charge-tree RHS evaluator with a GPU charge-tree
   traversal. Flatten charge-tree nodes, children, leaf charge indices, and
   multipole moments; keep those arrays on the GPU; launch one target thread
   per panel quadrature point; and retain direct evaluation only for accepted
   leaf charges. The fine 6CO8 run spends 312.12 s in `setupRHS`, now much more
   than its 41.74 s first FMM matvec.
5. Continue the fine-mesh size ladder beyond the completed two-unit case while
   retaining exact mesh reuse and the `1e-4` convergence requirement.
6. Estimate and test full-virus `R=1` GPU RHS separately. The new default GPU
   direct cap handles one-unit and large-PDB-sized cases, but full-virus `R=1`
   is still `1.61e13` panel-charge interactions and remains above the cap.
7. Avoid cached-LU (`-P=2`) on the first full-virus `R=1` staged tests because
   preconditioner setup starts before RHS and is already huge at 10.2M panels.
8. Repeat full-virus solves on preserved meshes and compare early residual
   histories for cached-LU and unpreconditioned GMRES under the same
   100-iteration cap.
9. For an exact same-mesh 1AON comparison, either run FMM against the preserved
   TABI-PB `triangulatedSurf.face/.vert` pair or add a TABI-PB mesh reuse path
   that reads an existing mesh instead of invoking NanoShaper.
10. Harden the FMM PQR parser so direct PDB2PQR fixed-width output is accepted
   without a compact intermediate file.
11. Do not interpret coarse or unconverged full-virus energies as approximations to the paper;
   first make a fine surface computationally feasible through GPU work and
   mesh/solution checkpointing.

## Reproduction Files

- Harness: `scripts/compare_tabi_fmm_size_sweep.sh`
- Zika subset generator: `scripts/extract_zikv_asymmetric_units.sh`
- Fine baseline: `results/comparison/tabi_fmm_size_sweep/20260801_baseline/summary.csv`
- Chain-A depth check: `results/comparison/tabi_fmm_size_sweep/20260801_chain_depth6/summary.csv`
- One-unit Arnoldi-100 result: `results/comparison/tabi_fmm_size_sweep/20260801_unit01_depth8_a100/summary.csv`
- Two-unit fine-mesh result: `results/comparison/tabi_fmm_size_sweep/20260802_unit02_sdens1/summary.csv`
- Full-virus R=2 attempt: `results/fmm/ZIKV_6CO8_zenodo_R2/fmm.log`
- Full-virus R=1 fine mesh: `results/fmm/ZIKV_6CO8_zenodo_R1_mesh/mesh.log`
- Full-virus R=1 GPU-request setup check: `results/fmm/ZIKV_6CO8_zenodo_R1_mesh/gpu_request_smoke.log`
- Local CUDA availability smoke check: `results/gpu_smoke/1a63_request/gpu_request.log`
- Local CUDA retest after GPU-kernel rebuild: `results/gpu_smoke/1a63_after_gpu_kernel_fix/gpu_request.log`
- Host GPU `1a63` one-iteration smoke: `results/gpu_smoke/1a63_host_gpu_after_fix/gpu_request.log`
- Host GPU `1a63` converged solve: `results/gpu_smoke/1a63_host_gpu_full/gpu.log`
- Matching CPU `1a63` converged solve: `results/gpu_smoke/1a63_host_cpu_full/cpu.log`
- Zika unit-one GPU tree-RHS solve: `results/gpu_large/ZIKV_6CO8_unit01_sdens1_p2_gpu/gpu.log`
- Zika unit-one GPU direct-RHS solve after policy update: `results/gpu_large/ZIKV_6CO8_unit01_sdens1_p2_gpu_auto_rhs/gpu.log`
- Full-virus R=1 interrupted `-P=2` GPU-stage attempt: `results/fmm/ZIKV_6CO8_zenodo_R1_mesh/gpu_stop_after_rhs_p2_interrupted.log`
- Full-virus R=1 no-preconditioner RHS-stage attempt: `results/fmm/ZIKV_6CO8_zenodo_R1_mesh/gpu_stop_after_rhs_nopc.log`
- Independent 1AON FMM convergence check: `results/fmm/1AON_charmm_protein_R1_p2/fmm.log`
- Independent 1AON TABI-PB result: `results/tabipb/1AON_charmm_protein_sdens1/tabipb.log`
- Additional large PDB batch: `results/comparison/tabi_fmm_large_pdbs/20260803_batch01/summary.csv`
- Resolution sweep: `results/comparison/tabi_fmm_resolution_sweep/summary.csv`
