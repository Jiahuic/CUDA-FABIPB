# setupRHS and GPU Tracking

Current branch:

```text
issue-zenodo-pqr-zero-radius
```

Problem:

The FMM matvec path is accelerated, but `setupRHS` still evaluates charge-to-panel
interactions directly before GMRES:

```text
work = surface panels * charge atoms
```

This is fine for small proteins. For the Zenodo 6CO8 scale-1 mesh:

```text
10,222,344 panels * 1,576,628 charges = 16,115,691,398,832 direct RHS pairs
```

Direction 1: avoid waiting on impossible direct RHS runs.

- The executable now has a direct-RHS guard.
- Default limit: `FABIPB_MAX_DIRECT_RHS_PAIRS=5000000000`.
- Override only for an intentional long run:

```sh
FABIPB_ALLOW_LARGE_DIRECT_RHS=1 ./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=0 -m=2 -R=1.0 -eps1=4 -eps2=80 test_proteins/ZIKV_6CO8_zenodo
```

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

Important limitation:

The current CUDA `setupRHS` implementation is also a direct all-pairs kernel.
It can measure GPU speedup for small/coarse cases, but it does not fix the
scale-1 6CO8 complexity. The method-level fix is an accelerated charge-to-panel
RHS evaluation, for example a charge-source FMM/treecode evaluated at panel
quadrature points.

Focused tests:

```sh
cmake --build build --parallel 8
tests/run_zenodo_issue_tests.sh
```

The wrapper runs:

- `tests/test_pqr_parser_zero_radius.sh`
- `tests/test_rhs_guard.sh`
- `tests/test_gpu_request_smoke.sh`
