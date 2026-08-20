# Viral Capsid GPU Performance Notes

Date: 2026-08-20

## Scope

This note records the recent FABIPB GPU changes and measured results for the
large capsid cases:

- ZIKV / 6CO8 Zenodo PQR: `test_proteins/ZIKV_6CO8_zenodo.pqr`
- H1N1 capsid PQR: `test_proteins/H1N1_atoms.pqr`

The goal is practical reproducibility: keep the calculation runnable at
paper-like mesh sizes, record which approximations are active, and identify the
remaining runtime bottlenecks.

## Current Production Defaults

The current runner is:

```sh
./scripts/run_6co8_fabipb_fast.sh <pqr>
```

Important defaults and options:

```text
mesh backend:                    NanoShaper
SDENS:                           1 unless set by environment
internal FMM R:                  1 / SDENS
dielectrics:                     PDIE=4, SDIE=80 for capsids
GMRES tolerance:                 1e-4
GMRES restart:                   30
preconditioner:                  auto; diagonal for huge capsids
Q2M/L2P GPU path:                auto; disabled for huge capsids by atom count
RHS tree theta:                  0.3
post-solve energy tree theta:    0.3
energy mode:                     panel-tree
GPU nearfield special cache:     65536 MiB source default
GPU mode:                        auto; use CUDA when available
worker threads:                  online CPU count
```

Set `FABIPB_ENERGY_TREE_THETA=0.2` to recover the stricter historical energy
tree setting. The current default is `0.3` because the capsid-scale accuracy
drift is acceptable for the present comparison work and the post-solve energy
kernel can otherwise dominate runtime.

The source also auto-selects Q2M/L2P unless `FMM_Q2M`/`-Q` is explicitly
overridden. GPU Q2M/L2P is useful for small and medium cases, but on huge
capsids it reserves GPU memory that nearfield streaming needs more. The current
automatic cutoff is `FABIPB_HUGE_CAPSID_ATOMS=5000000` atoms; the older
`FABIPB_Q2M_HUGE_CAPSID_ATOMS` name is still accepted as an alias.

The source auto-selects `FMM_PRECONDITIONER`/`-P` with the same huge-capsid
cutoff:

- huge capsids: `P=3` diagonal, because block-LU setup/apply cost and memory
  pressure are not worth the iteration tradeoff at this scale
- small/medium high-contrast cases with `eps2/eps1 >= 40` and `eps1 <= 2`:
  `P=2` cached block-LU, matching the measured `eps1=1` protein behavior
- all other cases: `P=3` diagonal

Set `FMM_PRECONDITIONER=<mode>` to override this policy for convergence
studies.

## Changes That Enabled These Runs

1. **GPU nearfield memory arbitration.** After GPU setupRHS, the RHS charge-tree
   cache is released before GMRES. This frees device memory for nearfield
   streaming tables and avoids CPU fallback on large capsids.
2. **Nearfield stream-pair and special caches.** The GPU nearfield path can keep
   a destination-leaf stream-pair table on device and cache special panel cases
   on host memory. For H1N1 `sdens=1`, this reduced nearfield from about
   `250 s` per matvec to about `36 s` per matvec.
3. **M2L streaming diagnostics.** M2L GPU fallback now reports the exact CUDA
   allocation/copy reason. Depths with too many flattened M2L pairs fail early
   with a clear message instead of overflowing indexing.
4. **GPU final energy reduction.** The post-solve charge-tree energy kernel now
   reduces panel contributions on the GPU by block partial sums. It copies back
   only one double per CUDA block instead of `2*nPanels` doubles and doing the
   full dot product on the CPU.
5. **Energy timing breakdown.** GPU energy logs now report build, scratch,
   `sgm` upload, kernel, partial copy, and reduction times.

## ZIKV / 6CO8 Results

### Reference Context

Paper / TABI-PB reference at `sdens=1`:

```text
vertices:          5,109,760
energy:           -117,632.1 kcal/mol
TABI-PB this run: -117,561.98 kcal/mol
```

FABIPB uses a panel-Galerkin discretization, so the energy is not expected to
match TABI-PB exactly at the same mesh density. The current target is stable,
trackable runtime with acceptable capsid-scale accuracy.

### Current `sdens=1`, depth 8, source defaults

Command:

```sh
SDENS=1 OUT_DIR=results/fmm/6co8_fabipb_sdens1/zenodo_sdens1_depth8_full ./scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

Recorded result:

```text
vertices:                 5,109,760
panels:                   10,222,292
GMRES:                    87 iterations
final residual:           9.865349e-05
solvation energy:         -123,425.979217 kcal/mol
total time:               722.240321 s
setupRHS:                 58.762306 s
GMRES:                    398.169222 s
treecode/energy:          227.098238 s
energy kernel:            223.492387 s
nearfield avg/call:       1.388252 s
```

Source-default rerun after moving Q2M, preconditioner, RHS theta, energy mode,
and thread/chunk choices into the binary:

```text
output directory:         results/fmm/6co8_fabipb_sdens1/zenodo_sdens1_depth8_full
vertices:                 5,109,760
panels:                   10,222,292
resolved Q2M:             1, auto small/medium case
resolved preconditioner:  3, auto diagonal
GMRES:                    87 iterations
final residual:           9.865349e-05
solvation energy:         -123,421.148457 kcal/mol
total time:               530.832819 s
setupRHS:                 59.640010 s
GMRES:                    323.349698 s
treecode/energy:          109.680781 s
energy kernel:            106.312824 s
nearfield avg/call:       1.391931 s
```

Previous comparable run before nearfield and energy improvements:

```text
GMRES:                    87 iterations
solvation energy:         -123,418.881588 kcal/mol
total time:               4,274.441313 s
GMRES:                    2,612.935698 s
treecode/energy:          1,496.200693 s
nearfield avg/call:       23.744000 s
```

Net result: same iteration count, about `5.9x` total speedup. The energy change
is about `7.1 kcal/mol`, approximately `0.006%`, consistent with changed GPU
reduction/order and tree evaluation details rather than a solver change.

## H1N1 Results

### Paper-Scale `sdens=0.5`, depth 8

Paper H1N1 table entry:

```text
vertices:                 12,313,670
GMRES iterations:         14
energy:                   -31,424,958.2 kcal/mol
reported time columns:    1112 s / 555 s
```

Current FABIPB `sdens=0.5` run with energy theta `0.2`:

```text
vertices:                 12,313,558
panels:                   24,740,712
GMRES:                    12 iterations
final residual:           9.314421e-05
solvation energy:         -32,504,698.501418 kcal/mol
total time:               1,219.954640 s
setupRHS:                 175.579148 s
GMRES:                    213.133154 s
treecode/energy:          696.819917 s
energy kernel:            690.616759 s
nearfield avg/call:       11.815454 s
```

Relative to the paper energy, this is about `3.44%` more negative, within the
current 3-6% acceptance band for large capsid comparisons.

After switching both RHS and post-solve energy tree defaults to `theta=0.3`,
the same old wrapper policy produced:

```text
output directory:         results/fmm/h1n1_fabipb_sdens05/test_default
vertices:                 12,313,558
panels:                   24,740,712
GMRES:                    12 iterations
final residual:           9.314421e-05
solvation energy:         -32,504,690.262723 kcal/mol
total time:               851.603530 s
setupRHS:                 174.769208 s
GMRES:                    216.350639 s
treecode/energy:          330.020849 s
energy kernel:            323.515543 s
nearfield avg/call:       11.967517 s
```

The solve completed successfully. The trailing `syntax error near unexpected
token f"Wrote ..."` in that run came from the shell script's post-run summary
block after FABIPB had already finished, not from the solver.

Before the recent GPU energy-reduction change, the comparable `sdens=0.5` run
was:

```text
total time:               4,027.339143 s
GMRES:                    2,895.369934 s
treecode/energy:          583.556946 s
energy:                   -32,504,575.639931 kcal/mol
```

The newer solve is much faster overall because GPU nearfield and GPU energy
paths are both active. The final energy remains a major cost at `theta=0.2`.

### `sdens=1`, depth 9

The `sdens=1` H1N1 case is much larger:

```text
vertices:                 51,402,202
panels:                   102,928,968
unknowns:                 205,857,936
```

Before releasing RHS charge-tree memory and enabling the nearfield special
cache, depth 9 often fell back to CPU nearfield or spent most time in
nearfield. The optimized full run:

```text
GMRES:                    15 iterations
final residual:           9.151024e-05
solvation energy:         -31,122,198.103054 kcal/mol
total time:               6,275.499151 s
setupRHS:                 1,030.447040 s
GMRES:                    1,056.391092 s
treecode/energy:          3,852.270190 s
nearfield avg/call:       36.043619 s
```

The same setup with default energy theta now set to `0.3` should reduce the
final energy cost. An accidental multiline shell run used `SDENS=1` despite the
user intending `SDENS=0.5`; it still provides a useful theta comparison:

```text
GMRES:                    15 iterations
final residual:           9.467684e-05
solvation energy:         -31,122,635.552882 kcal/mol
total time:               5,533.438084 s
GMRES:                    1,851.111582 s
treecode/energy:          2,711.264544 s
energy kernel:            2,693.883721 s
```

This confirms `theta=0.3` can reduce the H1N1 `sdens=1` energy stage, though
the multiline command also enabled `FMM_Q2M=1` and used `sdens=1`, so it should
not be treated as the canonical `sdens=0.5` paper-density run.

## Common Commands

H1N1 paper-density run with current defaults:

```sh
SDENS=0.5 OUT_DIR=results/fmm/h1n1_fabipb_sdens05/h1n1_sdens05_depth8_gpu_energy_default_theta03 ./scripts/run_6co8_fabipb_fast.sh test_proteins/H1N1_atoms.pqr
```

ZIKV / 6CO8 `sdens=1` run with current defaults:

```sh
SDENS=1 OUT_DIR=results/fmm/6co8_fabipb_sdens1/zenodo_sdens1_depth8_gpu_energy_default_theta03 ./scripts/run_6co8_fabipb_fast.sh test_proteins/ZIKV_6CO8_zenodo.pqr
```

Important shell rule: keep environment assignments on the same command line, or
use backslashes. A previous multiline command accidentally ran `SDENS=1` because
the final script invocation did not receive `SDENS=0.5`.

## Current Bottlenecks

After the nearfield and energy-reduction changes:

- For 6CO8 `sdens=1`, runtime is split between GMRES and final energy. Both
  are now practical.
- For H1N1 `sdens=0.5`, final energy still dominates at strict theta `0.2`.
- For H1N1 `sdens=1`, final energy and nearfield remain the major costs, but
  both run on GPU and are trackable.

The next optimization target is the screened charge-tree energy kernel itself.
The current GPU path is correct and avoids CPU fallback, but the screened
derivative tree walk is substantially more expensive than the Coulomb-only RHS
kernel.
