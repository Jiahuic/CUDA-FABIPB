# NanoShaper Integration Analysis

This note records the results of testing NanoShaper as a second surface mesher
alongside MSMS on the `nanoshaper-integration-experiment` branch. It covers
build requirements, mesh quality, convergence behavior across all test proteins,
resolution calibration, and the mechanism behind the GMRES iteration differences.

---

## Build

NanoShaper must be built from source in `nanoshaper/` against the system Boost.
The pre-built `nanoshaper-master/` binary requires `libboost_thread.so.1.74.0`
which is absent on Ubuntu 26.04 (Boost 1.90).

The blocking issue in `nanoshaper/CMakeLists.txt` was that `boost_system` is
listed as a required Boost component:

```cmake
find_package( Boost COMPONENTS thread system filesystem REQUIRED )
```

`boost_system` became header-only in Boost 1.69 and no longer ships a cmake
config file in Boost 1.90. It is not used anywhere in the NanoShaper source.
Remove `system` from the component list:

```cmake
find_package( Boost COMPONENTS thread filesystem REQUIRED )
```

The `cmake_minimum_required` / `project` call order was also wrong (fixed in the
same patch) and `cmake_policy(VERSION 2.8.4)` was removed as obsolete.

Build commands:

```sh
cmake -S nanoshaper -B nanoshaper/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCGAL_DIR=nanoshaper/cgal-5.6.2
cmake --build nanoshaper/build -- -j$(nproc)
```

`CMakeLists.txt` in the main project checks `nanoshaper/build/NanoShaper` first,
then falls back to `nanoshaper-master/build/NanoShaper`.

---

## What "stable mesher" means

NanoShaper's stability claim is about **surface generation robustness**: it
reliably produces a valid solvent-excluded surface for any protein, including
ones where MSMS crashes or produces holes due to complex topology. This is a
surface generation claim, not a solver convergence claim.

One hard limitation: NanoShaper requires at least 4 atoms. Its CGAL Delaunay
triangulation needs 4 non-coplanar points. The `oneb` test case (1 atom) will
always fail. The NanoShaper documentation suggests adding 3 dummy atoms with
zero radius at the same center as a workaround.

---

## Mesh quality

Both meshers approximate the same SES, but their triangulations differ
fundamentally. MSMS's rolling-probe algorithm produces cusp junction lines where
probe-sphere patches meet. These generate extremely elongated triangles at the
junctions that are filtered by `removePanelArtifacts` after reading; the
retained triangles still show high aspect ratio tails. NanoShaper smooths these
cusps via its CGAL-based algorithm.

Mesh quality measured across six representative proteins at `R=1.0`:

| Mesher | Median AR | p95 AR | Max AR | Area CV |
|--------|-----------|--------|--------|---------|
| MSMS   | 2.2–2.3   | 14–21  | >10¹²  | 0.83–0.88 |
| NanoShaper | 1.43–1.45 | 2.3 | 3.8–5.0 | 0.385–0.390 |

AR = max-edge / min-edge per triangle. Area CV = coefficient of variation of
panel areas. NanoShaper produces near-equilateral triangles and a uniform area
distribution across all proteins tested.

The degenerate MSMS triangles (AR > 10¹²) are the ones at cusp junctions.
They are counted in the raw `.vert`/`.face` output but filtered by the solver
before building the FMM tree (`removePanelArtifacts` in `src/input.c:596`).

---

## GMRES iteration comparison across all test proteins (R=1.0, CPU)

Negative delta means NanoShaper needed fewer iterations (faster convergence).

| Protein | MSMS its | NS its | Δ | MSMS panels | NS panels |
|---------|----------|--------|---|-------------|-----------|
| 1frd    | 10       | 25     | +15 | 11926     | 13180     |
| 1a63    | 17       | 29     | +12 | 20227     | 20744     |
| 1fca    | 12       | 24     | +12 | 6902      | 8316      |
| 1a2s    | 15       | 23     | +8  | 13180     | 13188     |
| 1neq    | 12       | 20     | +8  | 13929     | 14092     |
| 1sh1    | 14       | 19     | +5  | 7657      | 8096      |
| 1fxd    | 10       | 16     | +6  | 8226      | 9120      |
| 1svr    | 11       | 12     | +1  | 13992     | 13804     |
| 1bbl    | 13       | 13     | 0   | 7521      | 7780      |
| 1vjw    | 17       | 17     | 0   | 7812      | 8624      |
| 2erl    | 9        | 8      | −1  | 6506      | 6820      |
| 1cbn    | 10       | 8      | −2  | 6628      | 6916      |
| 1hpt    | 12       | 10     | −2  | 9392      | 9592      |
| 1mbg    | 11       | 9      | −2  | 9105      | 9092      |
| 1r69    | 11       | 9      | −2  | 8628      | 8952      |
| 2pde    | 11       | 9      | −2  | 6582      | 8076      |
| 1a7m    | 17       | 14     | −3  | 22491     | 22732     |
| 1ajj    | 11       | 8      | −3  | 6027      | 6404      |
| 1bpi    | 12       | 9      | −3  | 8831      | 9568      |
| 1ptq    | 13       | 10     | −3  | 8202      | 8584      |
| 1uxc    | 14       | 10     | −4  | 7772      | 8288      |
| 1vii    | 14       | 10     | −4  | 7342      | 7244      |
| 451c    | 23       | 17     | −6  | 12296     | 12368     |
| 1bor    | 16       | 9      | −7  | 8444      | 8508      |
| oneb    | 5        | FAIL   | —   | 88        | n/a       |

Summary: NanoShaper is better or equal on 16/24 proteins, worse on 8.

CPU/GPU `applyFMM` agreement is at machine epsilon (~5×10⁻¹⁶ rel_l2) for both
meshers on all proteins tested. GMRES iteration counts are identical between
CPU and GPU runs for both meshers.

---

## Why NanoShaper needs more iterations on specific proteins

The 8 proteins where NanoShaper is worse show a consistent structural pattern
in the preconditioner interaction histogram (`-B=1` output):

| Protein | Δits | MSMS non-disj% | NS non-disj% | Δ pp | NS near/call |
|---------|------|----------------|--------------|------|--------------|
| 1frd    | +15  | 39.8%          | 44.3%        | +4.5 | 0.93× MSMS   |
| 1a63    | +12  | 17.5%          | 19.6%        | +2.1 | 1.23×        |
| 1fca    | +12  | 55.9%          | 63.1%        | +7.2 | 1.06×        |
| 1a2s    | +8   | 37.7%          | 44.6%        | +6.9 | 1.31×        |
| 1neq    | +8   | 32.2%          | 37.7%        | +5.5 | 1.29×        |
| 1fxd    | +6   | 50.8%          | 56.4%        | +5.7 | 1.03×        |
| 1sh1    | +5   | 52.3%          | 60.5%        | +8.1 | 1.18×        |
| 1svr    | +1   | 35.6%          | 41.4%        | +5.8 | 1.27×        |

Non-disj% = fraction of preconditioner interactions that are touching or
adjacent (one-common + two-common + self), as reported by the
`Preconditioner panelIA0 cases` line. NS near/call = NanoShaper's near-field
time per matvec call relative to MSMS's.

**Mechanism.** MSMS cusp junctions pack many small panels into a compact
region that falls within a single octree leaf (preconditioner block). The block
LU factorization captures essentially all the operator energy for that cluster,
making the preconditioner locally near-exact. NanoShaper's uniform mesh has no
cusp clusters: panels are distributed evenly, so touching panels are spread
across block boundaries. The non-disjoint fraction within each block is higher
(more sharing of edges and vertices), but many of those touching interactions
cross block boundaries and are invisible to the preconditioner. The block
preconditioner misses more inter-block operator contributions, requiring more
GMRES iterations.

NanoShaper's cleaner triangles do make each near-field matvec call 3–31%
faster on these proteins (fewer expensive near-singular integrals). This
speedup does not compensate for 50–150% more iterations on the affected
proteins.

The effect is protein-specific. Proteins where MSMS cusp clusters do not
produce useful preconditioner blocks — either because the cusps are sparse or
because they fall across block boundaries — show no convergence disadvantage
from NanoShaper.

---

## Resolution behavior and calibration

The `-R` control maps differently across backends:

- MSMS: `density = 1 / R²`
- NanoShaper: `Grid_scale = 1 / R`

At `R=1.0` the panel counts are close (within ~2–6% for most proteins).
At `R>1.25` the mappings diverge: NanoShaper produces substantially fewer
panels than MSMS at the same nominal `-R`, and the surface loses area accuracy.
At `R=2.0` on 1a63, NanoShaper produces 4,980 panels (MSMS: 14,361) and
gives a physically unreliable energy.

Resolution sweep on 1a63:

| R    | MSMS panels | MSMS its | MSMS energy  | NS panels | NS its | NS energy    |
|------|-------------|----------|--------------|-----------|--------|--------------|
| 0.75 | 28601       | 13       | −2527.38     | 37236     | 28     | −2536.98     |
| 1.00 | 20227       | 17       | −2756.26     | 20744     | 29     | −2685.49     |
| 1.25 | 17071       | 13       | −3150.48     | 13204     | 29     | −2910.76     |
| 1.50 | 14967       | 13       | −4124.09     | 9004      | 26     | −3350.46     |
| 2.00 | 14361       | 13       | −4262.30     | 4980      | 20     | −10099.88 ⚠  |

**Use `R≤1.25` for NanoShaper.** For cross-mesher comparisons at matched panel
counts, use the calibration script:

```sh
RESOLUTIONS="0.75 1.00 1.25" ./scripts/calibrate_mesh_resolution.sh test_proteins/1a63
```

`R=1.0` is the canonical comparison point: panel counts match within 2–6% for
most proteins in the test set.

---

## GPU results at R=1.0 on 1a63

| Path            | Time    | GMRES its | Energy      |
|-----------------|---------|-----------|-------------|
| MSMS CPU        | 6.59 s  | 17        | −2756.26    |
| MSMS GPU        | 0.86 s  | 17        | −2756.24    |
| NanoShaper CPU  | 8.46 s  | 29        | −2685.49    |
| NanoShaper GPU  | 0.94 s  | 29        | −2685.60    |

MSMS GPU speedup: ~7.7×. NanoShaper GPU speedup: ~9.0×. The larger NS speedup
reflects faster per-call near-field on GPU with the cleaner triangulation.

---

## Comparison workflow

Mesh-only test:

```sh
./build/fabipb -m=2 -M=1 -R=1.0 test_proteins/1cbn
```

CPU/GPU solve comparison with NanoShaper:

```sh
MESH_MODE=2 ./scripts/compare_gpu_cpu.sh test_proteins/1ajj
```

CPU/GPU correctness check:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=1 -c=1 -m=2 -R=1.0 test_proteins/1ajj
```

Cross-mesher calibration:

```sh
RESOLUTIONS="0.75 1.00 1.25" ./scripts/calibrate_mesh_resolution.sh test_proteins/1a63
```
