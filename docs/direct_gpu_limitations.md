# Direct GPU Baseline Limitations

## Purpose

This note explains why the current direct GPU baseline (`-r=1`) is only
practical for small cases and why GPU-FMM integration is necessary for realistic
PB problem sizes.

## What the current direct GPU path does

The current direct GPU baseline is a dense all-pairs panel interaction path.

For `nPnls` surface panels, it builds all `nPnls * nPnls` panel-panel
interactions and stores four double-precision coefficients per pair:

- `k0`
- `k1`
- `k2`
- `k3`

These are the four coefficients returned by `panelIA0(pnlX, pnlY)` for each
source-destination panel pair.

So the memory cost scales as:

- `O(nPnls^2)` interactions
- `4 * sizeof(double)` bytes per interaction just for the coefficients

This is a dense cached operator, not an on-the-fly direct solver.

## Why large cases fail

On a large case such as `1a63` at MSMS density `10`, the direct GPU path requires
hundreds of gigabytes of device memory.

Example observation from the benchmark machine:

- `#ele = 132196`
- estimated direct GPU device memory: about `520 GB`
- available GPU memory: about `23 GB`

So the current dense direct GPU path cannot run. It falls back to FMM.

This is why a run like:

```sh
./scripts/with_benchmark_env.sh ./build/fabipb -B=1 -g=1 -r=1 -m=1 -R=0.5 test_proteins/1a63
```

prints:

- `Direct GPU cache unavailable ...`
- `Direct GPU matvec unavailable; using FMM path.`

and then ends up measuring the FMM solver instead of a true direct-sum solve.

## Why Near/FMM timings disappear in that mode

When `-r=1` is requested, the code labels the run as direct-baseline mode.
Current benchmark printing suppresses the usual FMM stage summary in that mode.

So if the direct path falls back to FMM:

- the actual matvec is FMM
- but the end-of-run print still omits FMM stage totals

This makes the run look like a direct-sum run even though it is not.

This is a reporting issue, not a mathematical one.

## Why this supports GPU-FMM integration

The dense direct GPU path hits the expected `O(n^2)` memory wall.

That is exactly why GPU-FMM integration matters:

- FMM avoids storing the full dense operator
- far-field interactions are compressed
- only near-field interactions are treated directly
- realistic PB problems remain feasible on current GPUs

So the current direct baseline is best viewed as:

- a reference implementation for small cases
- not a scalable production path

## Future direct-GPU direction

If we want a more meaningful direct GPU comparison for larger problems, the next
design should not store the full dense matrix.

Two better alternatives are:

### 1. Row-block / destination-block direct GPU

- process a block of destination panels at a time
- stream source panels across that block
- accumulate the result directly
- discard temporary coefficient data after each block

This reduces memory usage from full `O(n^2)` storage to a block-sized working
set.

### 2. On-the-fly direct GPU

- do not store panel-pair coefficients at all
- compute the quadrature interaction on the GPU as needed

This has much lower memory usage, but it is much harder because the PB panel
integrals are more complex than a simple point-particle kernel.

## Current recommendation

For the current paper and benchmark workflow:

- use the in-tree direct GPU baseline only on cases that fit in memory
- use it as a small-case reference, not a large-case production baseline
- emphasize that large-scale PB performance requires GPU-FMM integration

The direct GPU baseline is still useful, but only within its memory-feasible
regime.
