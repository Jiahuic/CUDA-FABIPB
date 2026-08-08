# setupRHS and GPU Tracking

Current branch:

```text
issue-zenodo-pqr-zero-radius
```

Dataset:

- Zenodo record: https://zenodo.org/records/4568768
- DOI: https://doi.org/10.5281/zenodo.4568768
- Archive: `pqr.zip`, PQR file: `ZIKV_6CO8_aa_charge_vdw_addspace.pqr`
- Fetch locally with `./scripts/fetch_zikv_6co8_zenodo.sh`, which writes
  `test_proteins/ZIKV_6CO8_zenodo.pqr`.

Problem:

The FMM matvec path is accelerated, but `setupRHS` still evaluates charge-to-panel
interactions directly before GMRES:

```text
work = surface panels * charge atoms
```

This is fine for small proteins. For the Zenodo 6CO8 scale-1 mesh:

```text
10,222,292 panels * 1,576,628 charges = 16,116,751,791,376 direct RHS pairs
```

Direction 1: use the accelerated charge-tree RHS.

- Pair counts above the direct limit automatically use the charge-tree RHS.
- Default limit: `FABIPB_MAX_DIRECT_RHS_PAIRS=5000000000`.
- The charge tree uses at least fourth-order expansions and a default acceptance
  ratio of `FABIPB_RHS_TREE_THETA=0.2`.
- `FABIPB_FORCE_TREE_RHS=1` selects the tree path for small reference cases.
- `FABIPB_DEBUG_COMPARE_RHS=1` compares it with the direct RHS.
- Override only for an intentional long run:

```sh
FABIPB_ALLOW_LARGE_DIRECT_RHS=1 ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=1.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

Validated `1a63` scale-1 result:

```text
panels:                       20,744
potential RHS relative L2:   0.00647%
normal RHS relative L2:      0.01763%
direct energy:              -2685.486955 kcal/mol
tree-RHS energy:            -2685.488928 kcal/mol
energy difference:           0.0000735%
GMRES iterations:            29 for both paths
```

RHS equation setup check:

`FABIPB_DEBUG_RHS_NORMS=1` compares the assembled root-FMM RHS against the
TABI-PB source-term formula evaluated at root's own panel quadrature points,
after dividing the integrated Galerkin RHS by panel area. On `1a63` with
NanoShaper `R=8`, `eps1=4`, `eps2=80`, direct RHS matches to roundoff:

```text
potential relative L2:       2.83e-15
normal derivative rel L2:    4.24e-15
```

For the tree-accelerated RHS on the same case:

```text
potential relative L2:       5.39e-05
normal derivative rel L2:    3.83e-05
```

Matched `1a63` TABI-PB/root comparison:

TABI-PB `usrdata.in` uses `sdens=2`, `pdie=1`, `sdie=80`. The matching root
NanoShaper setting is `-R=0.5` because root maps `R` to
`Grid_scale = 1/R`.

```text
TABI-PB command:
  cd TABI-PB/examples
  ../build/bin/tabipb usrdata.in

mesh:                    42,260 vertices, 84,512 triangles
area:                    6,931.47
GMRES:                   12 iterations, final-residual=7.929615e-05
solvation energy:       -2,443.836782 kcal/mol
```

Root mesh-only check:

```text
command:
  ./scripts/with_benchmark_env.sh ./build/fabipb \
    -B=1 -g=0 -m=2 -R=0.5 -eps1=1 -eps2=80 -M=1 \
    TABI-PB/examples/1a63

mesh:                    42,260 vertices, 84,512 panels
area:                    6,931.478245
```

Root panel-Galerkin result from the matched run:

```text
command:
  ./scripts/with_benchmark_env.sh ./build/fabipb \
    -g=0 -m=2 -R=0.5 -eps1=1 -eps2=80 \
    TABI-PB/examples/1a63

GMRES:                   58 iterations
solvation energy:       -2,480.912321 kcal/mol
delta vs TABI-PB:        about 1.5%
```

Root TABI-style nodepatch-tree diagnostics on the same mesh:

```text
theta=0.8, leaf_max=256, charge_theta=0.3
GMRES:                   13 iterations, final-residual=7.730901e-05
root nodepatch-tree:    -2,573.432051 kcal/mol

theta=0.6, leaf_max=128, charge_theta=0.2
GMRES:                   13 iterations, final-residual=9.926043e-05
root nodepatch-tree:    -2,522.058750 kcal/mol
```

For `1a63`, the existing root panel-Galerkin path is already close to TABI-PB,
while the approximate nodepatch-tree diagnostic is within a few percent and
trends toward TABI-PB as the tree setting is tightened. This is the opposite
pattern from 6CO8, where root nodepatch-tree is close to TABI-PB and the
panel-Galerkin path is far away.

On the Zenodo 6CO8 PQR with an exact reused coarse NanoShaper `R=128` mesh
(`72` panels, `1,576,628` charges, `eps1=4`, `eps2=80`), direct RHS also
matches the quadrature-equivalent TABI-PB source formula to roundoff:

```text
potential relative L2:       5.08e-14
normal derivative rel L2:    4.99e-13
```

For the tree-accelerated RHS on that same reused 6CO8 mesh:

```text
potential relative L2:       4.56e-06
normal derivative rel L2:    1.93e-06
```

For the medium 6CO8 mesh (`20,988` panels and `1,576,628` charges), the
fourth-order tree RHS with acceptance ratio `0.2` completes in about `11.0 s`.
The equivalent direct calculation contains `33,090,268,464` panel-charge pairs.

GMRES and operator diagnosis:

- `-i=<n>` controls the GMRES maximum iteration count; every run now reports
  `info`, iterations, and final residual.
- `FABIPB_REUSE_MESH=1` preserves and reuses `.face/.vert` files so direct and
  FMM operators can be compared on exactly the same mesh. Reuse only with the
  same mesh backend and resolution that created the files.
- On one frozen 72-panel 6CO8 mesh with preconditioning disabled, direct and
  FMM both converged in 118 iterations. Their energies differed by `0.126%`.
- The default cached-LU preconditioner did not converge after 200 iterations
  on that mesh (`final-residual=9.365274e-01`), while the unpreconditioned
  direct solve converged in 118 iterations (`final-residual=8.200292e-05`).
- The medium 20,988-panel FMM solve with preconditioning disabled converged in
  242 iterations (`final-residual=9.987312e-05`). Its energy,
  `8,493,420.369490 kcal/mol`, is still physically invalid because this mesh is
  much too coarse to reproduce the scale-1 virus surface and paper result.
- Repeating that medium Zenodo 6CO8 solve with the current tree-RHS path and
  same-panel operator terms enabled gives the same conclusion:

  ```text
  command:
    FABIPB_FORCE_TREE_RHS=1 ./scripts/with_benchmark_env.sh ./build/fabipb \
      -B=1 -g=0 -m=2 -R=16.0 -eps1=4 -eps2=80 -P=-1 -i=300 \
      test_proteins/ZIKV_6CO8_zenodo

  mesh:                    20,988 panels, area=1,587,532.219130
  RHS mode:                tree, theta=0.2
  GMRES:                   242 iterations, final-residual=9.987605e-05
  solvation energy:        8,493,393.695798 kcal/mol
  ```

  Tightening only the charge-tree RHS acceptance from `0.2` to `0.1` did not
  materially change the result:

  ```text
  command:
    FABIPB_FORCE_TREE_RHS=1 FABIPB_RHS_TREE_THETA=0.1 \
      ./scripts/with_benchmark_env.sh ./build/fabipb \
      -B=1 -g=0 -m=2 -R=16.0 -eps1=4 -eps2=80 -P=-1 -i=300 \
      test_proteins/ZIKV_6CO8_zenodo

  GMRES:                   242 iterations, final-residual=9.987601e-05
  setupRHS time:            38.321843 s
  solvation energy:        8,493,393.404090 kcal/mol
  energy delta vs theta=0.2: -0.291708 kcal/mol
  ```

  This rules out tree-RHS acceptance error as the source of the large
  virus-scale energy discrepancy at this mesh.
- A root run matching TABI-PB's coarse `sdens=0.1` NanoShaper setting uses
  `-R=10.0` (`Grid_scale=0.1`) on the Zenodo PQR:

  ```text
  command:
    FABIPB_FORCE_TREE_RHS=1 ./scripts/with_benchmark_env.sh ./build/fabipb \
      -B=1 -g=0 -m=2 -R=10.0 -eps1=4 -eps2=80 -P=-1 -i=300 \
      test_proteins/ZIKV_6CO8_zenodo

  mesh:                    64,024 panels, area=1,870,811.056459
  RHS mode:                tree, theta=0.2
  GMRES:                   244 iterations, final-residual=9.949363e-05
  solvation energy:        2,272,892.948366 kcal/mol
  ```

  TABI-PB rerun on the same Zenodo PQR with `sdens=0.1` produced the same
  mesh scale as root `R=10`:

  ```text
  command:
    cd TABI-PB/examples
    ../build/bin/tabipb 6co8_zenodo_r10.in

  mesh:                    31,056 vertices, 64,024 triangles
  area:                    1,870,810.792933
  GMRES:                   32 iterations, final-residual=9.171480e-05
  solvation energy:       -1,450,973.574187 kcal/mol
  ```

  Root's panel-Galerkin FMM path and TABI-PB's nodepatch collocation path are
  therefore using aligned PQR and mesh counts at this setting, but the panel
  result remains far away:

  ```text
  root panel-Galerkin FMM: +2,272,892.948366 kcal/mol
  TABI-PB nodepatch:      -1,450,973.574187 kcal/mol
  ```

  A new root diagnostic path builds TABI-style nodepatch unknowns from the same
  NanoShaper vertices and applies an approximate nodepatch tree matvec. This is
  enabled with `FABIPB_DEBUG_TABI_NODEPATCH=1` and
  `FABIPB_TABI_NODEPATCH_TREE=1`.

  First full `R=10` nodepatch-tree checks:

  ```text
  command:
    FABIPB_REUSE_MESH=1 FABIPB_DEBUG_TABI_NODEPATCH=1 \
    FABIPB_STOP_AFTER_TABI_NODEPATCH=1 FABIPB_TABI_NODEPATCH_TREE=1 \
    FABIPB_TABI_NODEPATCH_TREE_THETA=0.8 \
    FABIPB_TABI_NODEPATCH_TREE_LEAF_MAX=256 \
    FABIPB_TABI_NODEPATCH_CHARGE_TREE_THETA=0.3 \
    FABIPB_TABI_NODEPATCH_MAX_ITER=50 \
    ./scripts/with_benchmark_env.sh ./build/fabipb \
      -B=1 -g=0 -m=2 -R=10.0 -eps1=4 -eps2=80 -P=-1 \
      test_proteins/ZIKV_6CO8_zenodo

  GMRES:                   32 iterations, final-residual=9.276642e-05
  root nodepatch-tree:    -1,478,963.164764 kcal/mol
  ```

  Tighter tree settings gave the same scale:

  ```text
  theta=0.6, leaf_max=128, charge_theta=0.2
  GMRES:                   32 iterations, final-residual=8.965247e-05
  root nodepatch-tree:    -1,481,549.874416 kcal/mol
  ```

  These root nodepatch-tree values are within about `2%` of the TABI-PB rerun
  and have the same sign, while the panel-Galerkin FMM path does not. Current
  conclusion: the large 6CO8 discrepancy is primarily the equation
  discretization path, not the Zenodo PQR, mesh generation, raw kernel formulas,
  setupRHS, or FMM approximation.

## Galerkin vs. collocation: is the discretization gap inherent or fixable?

Follow-up to the "discretization path" conclusion above: isolate whether the
gap is an inherent property of Galerkin vs. collocation as paradigms, or a
fixable quadrature-order parameter, using a cheap single-subunit test case.

`test_proteins/ZIKV_6CO8_chainA_monomer.pqr` is one ~501-residue subunit
extracted from chain A of the Zenodo PQR (chain A in that file is ~60
repeated copies of the same subunit from icosahedral symmetry; residue
numbering resets to 1 every 501 atoms, so the first copy was cut out). It is
7,534 atoms, compact and non-hollow (unlike the full capsid), and solves in
seconds to low minutes instead of hours, making it a much cheaper
reproduction case for both the mesh-resolution and the preconditioner
questions.

Root panel-Galerkin FMM resolution sweep on this monomer (`-P=2` default,
`-q=1` default, converged GMRES only) against matched-resolution TABI-PB:

```text
R=8   (668 panels,     area=12,028): root=   +491.0
R=4   (3,320 panels,   area=15,263): root=-14,267.2
R=2   (14,820 panels,  area=17,914): root= -3,245.1
R=1   (62,972 panels,  area=19,826): root=   -831.9   vs TABI sdens=1 (area=19,826): -788.2  (5.5%)
R=0.5 (255,076 panels, area=20,860): root=   -745.5   vs TABI sdens=2 (area=20,860): -740.1  (0.72%)
```

TABI-PB itself is not fully converged at sdens=1/2 either (sdens=1→2→4 gives
-788.2 → -740.1 → -728.3 kcal/mol); Aitken-extrapolating that sequence gives
an estimated converged value of about `-724.4` kcal/mol.

**This confirms root's panel-Galerkin path converges to TABI-PB's answer
given enough resolution** (0.72% at matched fine mesh) -- ruling out a
correctness bug in the discretization itself, and reframing the large-6CO8
gap as primarily a resolution/convergence-rate difference between the two
methods (root needs a much finer mesh than TABI-PB to reach the same
accuracy on this geometry).

Quadrature-order re-sweep at the same resolutions, `-q=3` instead of default
`-q=1` (same mesh, `FABIPB_REUSE_MESH=1`, everything else unchanged):

```text
R=8  (668 panels):    q=1=+491.0    q=3=+1,235.0   (worse -- mesh too coarse for order to help)
R=4  (3,320 panels):  q=1=-14,267.2 q=3=  -409.2   (error vs -724.4: 1870% -> 43.5%)
R=2  (14,820 panels): q=1= -3,245.1 q=3=-1,061.0   (error vs -724.4:  348% -> 46.5%)
R=1  (62,972 panels): q=1=  -831.9  q=3=  -769.5   (error vs -724.4: 14.8% ->  6.2%)
```

Command pattern:

```sh
FABIPB_REUSE_MESH=1 ./scripts/with_benchmark_env.sh ./build/fabipb \
  -B=1 -g=0 -m=2 -R=4.0 -eps1=4 -eps2=80 -q=3 -i=300 \
  test_proteins/ZIKV_6CO8_chainA_monomer
```

At every resolution except the coarsest (R=8, where the mesh itself is too
coarse to represent the geometry regardless of quadrature accuracy), raising
quadrature order from the default 1 to 3 roughly halves or better the
relative error. `q=3` at R=4 (3,320 panels, 43.5% error) is already more
accurate than `q=1` at R=1 (62,972 panels, 14.8% error) -- a ~19x reduction
in panel count for comparable accuracy, from a CLI parameter, not a mesh
change. Quadrature cost scales as `qOrder^4` per near-field pair, so this is
not free (`-q=10` on the R=10 6CO8 capsid mesh burned 6.8 CPU-hours without
finishing setup and had to be killed), but `-q=3` is tractable and already
gave a 4x error reduction on the full R=10 6CO8 capsid mesh in a prior test
(`+2,272,893` -> `+549,632` kcal/mol, same mesh, same charges).

Panel-case ablation at R=4 (`FABIPB_SKIP_PANEL_CASES=self|one|two`) to
localize which touching-panel category drives the order sensitivity:

```text
              q=1 energy    q=3 energy    q1->q3 shift
none          -14,267.2       -409.2         +13,858
skip self     -22,453.2     -1,328.0         +21,125
skip one      -15,486.9       -408.2         +15,079
skip two      -15,276.3       -390.6         +14,886
```

No single category cleanly absorbs the order-sensitivity when excluded --
removing self, one-common, or two-common pairs all still show a large
(13,858-21,125) q1->q3 swing. The sensitivity is distributed across all
three touching/self categories, not isolated to one of `pnlThr0`/`pnlOne0`/
`pnlTwo0`.

A sharper localization comes from the existing `FABIPB_DEBUG_OPERATOR_NORMS=1`
diagnostic (`compareOperatorToTabiCollocation` in `src/fabipb.c`), which
already reimplements TABI-PB's exact point-collocation formula and diffs it
per-kernel-block against root's own operator (via the direct, non-FMM-
approximated `cpuDirectApply`), applied to the constant "ones" vector so no
RHS/charges are involved -- a pure discretization-formula comparison on the
same mesh:

```sh
FABIPB_REUSE_MESH=1 FABIPB_DEBUG_OPERATOR_NORMS=1 FABIPB_OPERATOR_VECTOR=ones \
  FABIPB_STOP_AFTER_OPERATOR=1 ./scripts/with_benchmark_env.sh ./build/fabipb \
  -B=1 -g=0 -m=2 -R=4.0 -eps1=4 -eps2=80 -q=3 test_proteins/ZIKV_6CO8_chainA_monomer
```

Results on the R=4 monomer mesh (relative L2 disagreement between root's
operator and the TABI-style formula, per kernel block):

```text
block  q=1 rel_l2   q=3 rel_l2   trend
k0        0.74%        0.07%     10x better
k1       29.3%        20.7%      improves, but stays large
k2       12.0%         8.1%      improves moderately
k3        6.3%         9.5%      gets worse
```

`k1` (the `epsilon*dGk/dy - dG0/dy` term) is the dominant, persistent source
of disagreement between root's Galerkin operator and TABI-PB's collocation
formula -- it is the largest gap at both orders and barely closes with more
quadrature points, unlike `k0` which converges almost immediately. `k3`
(the double-layer-difference term) gets slightly *worse* at higher order, a
smaller but real anomaly worth further investigation, not noise.

**Revised conclusion:** the large-6CO8 gap is not purely an inherent,
unavoidable property of Galerkin vs. collocation as discretization
paradigms. Quadrature order is a real, substantial, and currently
under-exploited lever (root's default is `-q=1`) that closes most of the gap
once the mesh is at least moderately resolved -- but the sensitivity is
concentrated disproportionately in the `k1` kernel block rather than spread
evenly across panel-pair categories, which is the next thing to chase if
pushing this further.

Separately, on the `issue-precond-capsid-divergence` branch: the default
cached-LU near-field preconditioner (`-P=2`) diverges on the full 6CO8
capsid mesh independent of both charge count and panel count (confirmed via
mesh/charge decoupling tests), and TABI-PB's own comparable per-leaf
block-LU preconditioner (opt-in, off by default) crashes on the same mesh --
this is a preconditioner-specific failure, orthogonal to the discretization
question above. A new `-P=3` diagonal/Jacobi preconditioner (mirroring
TABI-PB's default) converges cleanly (19 iterations) on the exact
full-charge capsid mesh where `-P=2` never converges.

Direction 2: test current GPU path.

The binary can be built with CUDA, and `-g=1` requests GPU execution:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=1 -m=2 -R=8.0 test_proteins/1a63
./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=1 -m=2 -R=128.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

On the current local machine, `nvidia-smi` cannot communicate with an NVIDIA
driver, and a `-g=1` run reports:

```text
CUDA backend unavailable: no CUDA-capable device is detected
```

So GPU timing must be collected on a GPU machine.

GPU limitation:

The current CUDA `setupRHS` implementation is also a direct all-pairs kernel.
It can measure GPU speedup for small/coarse cases, but the CPU charge-tree path
is currently the scalable `setupRHS` implementation.

## Fine 6CO8 RHS Acceleration Check

The first-iteration fine-mesh GPU comparison completed with:

```text
panels:                 10,222,264
charges:                 1,576,628
CPU charge-tree RHS:       312.119 s
first GPU FMM matvec:        41.737 s
GPU nearfield:               24.689 s
```

The RHS is therefore the dominant stage. It already uses 72 pthread workers,
one quadrature point per panel (`qOrd=1`), and a fourth-order charge expansion.
The remaining work is approximately 10.2 million independent recursive charge
tree walks. The existing direct CUDA RHS is not appropriate here because the
full interaction count is 16,116,707,645,792.

A same-mesh `1a63` sweep measured the CPU tree acceptance-ratio tradeoff:

| theta | setupRHS | potential rel-L2 error | normal rel-L2 error |
|---:|---:|---:|---:|
| 0.20 | 0.0754 s | 6.47e-5 | 1.76e-4 |
| 0.25 | 0.0583 s | 1.93e-4 | 6.63e-4 |
| 0.30 | 0.0470 s | 5.10e-4 | 2.39e-3 |
| 0.40 | 0.0339 s | 1.80e-3 | 1.07e-2 |

`theta=0.3` is a useful explicit 6CO8 timing experiment, but it is not the new
default until its fine-virus RHS sums are compared against the existing
`theta=0.2` result. The expected long-term implementation is a flattened GPU
charge tree: persistent node/moment/leaf-charge arrays, one GPU target per
panel quadrature point, an explicit bounded traversal stack, multipole
evaluation for accepted nodes, and direct charge evaluation only in reached
leaves. This changes the hardware used without changing the current tree-RHS
equations or acceptance rule.

Focused tests:

```sh
cmake --build build --parallel 8
tests/run_zenodo_issue_tests.sh
```

The wrapper runs:

- `tests/test_pqr_parser_zero_radius.sh`
- `tests/test_rhs_guard.sh`
- `tests/test_rhs_tree_accuracy.sh`
- `tests/test_gpu_request_smoke.sh`
