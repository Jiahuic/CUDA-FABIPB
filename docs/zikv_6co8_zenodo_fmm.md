# ZIKV 6CO8 Zenodo Input for Root FMM

This note tracks the root FMM parser issue found with the Zenodo ZIKV 6CO8 PQR.

Dataset:

- Zenodo record: https://zenodo.org/records/4568768
- DOI: https://doi.org/10.5281/zenodo.4568768
- Archive: `pqr.zip`
- PQR file: `ZIKV_6CO8_aa_charge_vdw_addspace.pqr`

Prepare the local input:

```sh
./scripts/fetch_zikv_6co8_zenodo.sh
```

The script writes:

```text
test_proteins/ZIKV_6CO8_zenodo.pqr
```

Expected PQR stats:

```text
atoms=1576628
mesh_atoms=1558076
zero_radius=18552
```

Root FMM parser behavior:

- PQR coordinates, charge, and radius are read from the last five numeric tokens.
- This supports both legacy PQR rows without a chain column and Zenodo rows with a chain column.
- Zero-radius atoms are retained in `sys->pos/sys->chr` as charges.
- Only positive-radius atoms are written to `.xyzr` for surface generation.

Smoke tests:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -g=0 -m=2 -R=1.0 -M=1 test_proteins/ZIKV_6CO8_zenodo
./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=128.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

The scale-1 mesh-only test should report about 5,109,790 vertices and 10,222,344
faces/root-FMM panels. The full `R=128` run is only a code-path smoke test; its
energy is not physically useful.

Known remaining limitation:

The exact scale-1 full solve is not practical in the current root FMM path because
`setupRHS` still computes direct panel-atom interactions before GMRES.

For the scale-1 Zenodo mesh this direct RHS work is approximately:

```text
10,222,344 panels * 1,576,628 charges = 16,115,691,398,832 panel-charge pairs
```

The executable has a guard for this direct path. By default, runs above
`FABIPB_MAX_DIRECT_RHS_PAIRS=5000000000` stop before entering `setupRHS`.
Override only for an intentional long benchmark:

```sh
FABIPB_ALLOW_LARGE_DIRECT_RHS=1 ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=1.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

GPU direction:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=1 -m=2 -R=128.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

The current CUDA `setupRHS` path is also direct over panel-charge pairs. It is
useful for measuring GPU speedup on smaller/coarser cases, but it is not the
final fix for the scale-1 virus case. The needed method change is an accelerated
charge-to-panel RHS evaluation.
