# GPU Backend Explained

This note explains how the current CUDA backend in
[`src/gpu_backend_cuda.cu`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/gpu_backend_cuda.cu)
accelerates the solver.

The implementation is not a generic GPU wrapper. It is a stage-by-stage GPU
backend for the FMM matvec and a direct dense GPU baseline.

## Overview

The CUDA backend currently accelerates four solver components:

1. near-field direct interactions
2. `M2L` far-field translations
3. leaf `Q2M`
4. leaf `L2P`

It also contains a separate direct dense GPU baseline for non-FMM comparisons.

The core design pattern is:

- build geometry-dependent coefficient tables once on the host
- flatten them into contiguous arrays
- upload them to device memory
- reuse them across repeated GMRES matvec calls

This is why the code uses persistent cache objects instead of recomputing GPU
data every iteration.

## Cache Objects

The file defines four main cache structures.

### `NearfieldGpuCache`

Defined near the top of
[`src/gpu_backend_cuda.cu`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/gpu_backend_cuda.cu).

This cache stores:

- flattened source panel indices
- flattened destination panel indices
- grouped leaf metadata
- four near-field coefficients per panel pair: `k0`, `k1`, `k2`, `k3`
- device buffers for `sgm` and `pot`

These coefficients come from CPU calls to `panelIA0()`, but the repeated
accumulation is done on GPU.

### `M2LGpuCache`

This cache stores the far-field interaction data for grouped `M2L`.

It includes:

- flattened M2L source cube indices
- destination-group metadata
- coefficient offsets per M2L pair
- flattened derivative tables `G0` and `Gk`
- flattened cube moment/local-expansion buffers
- indexing tables needed to reproduce the Cartesian `convM2L()` ordering on GPU

This is the GPU implementation of the cluster-to-cluster far-field interaction.

### `LeafTransformGpuCache`

This cache stores finest-level leaf transforms for:

- `Q2M0`, `Q2M1`
- `L2P0`, `L2P1`

It also keeps flattened moment and local-expansion buffers for the leaf-level
GPU transforms.

### `DirectGpuCache`

This is the dense direct baseline path.

It stores all panel-pair interaction coefficients for the full dense operator,
without using the FMM hierarchy.

It exists for benchmark/reference comparisons, not as the scalable production
path.

## GPU Availability

The runtime entry point is:

- `gpuBackendAvailable()`

This calls `cudaGetDeviceCount()` and returns whether a usable CUDA device is
visible. If not, the solver falls back to CPU.

## Near-Field Acceleration

### Host build

The near-field cache is built by:

- `buildNearfieldTables(const ssystem *sys)`

This function:

1. walks the flattened near leaf-pair list
2. computes how many panel-pair interactions exist
3. allocates flat host arrays
4. computes `panelIA0()` coefficients for every near panel pair
5. uploads the flattened tables to the GPU

The expensive part of the build is coefficient generation:

- `panelIA0(pnlX, pnlY)`

That step still happens on CPU, but the results are reused across matvecs.

### Kernels

Two near-field kernels exist:

- `nearfieldApplyKernel`
- `nearfieldLeafApplyKernel`

`nearfieldApplyKernel` is the older interaction-based path:

- one thread handles one panel-pair interaction
- contributions are accumulated with `atomicAdd`

`nearfieldLeafApplyKernel` is the grouped path now used by default:

- one CUDA block owns one destination leaf
- threads iterate over destination panels inside that leaf
- all source interactions contributing to that destination leaf are accumulated
  locally

This destination-leaf grouping is the main near-field innovation in the current
codebase.

### Runtime entry point

Near-field application is called through:

- `gpuNearfieldApply(ssystem *sys, double alpha, const double *sgm, double *pot)`

This function:

1. builds the cache the first time a system is seen
2. copies `sgm` and `pot` to the device
3. launches the grouped or interaction kernel
4. copies `pot` back to the host

The code also records separate timings for:

- build
- host-to-device copy
- kernel
- device-to-host copy

## `M2L` Acceleration

### Host build

The grouped `M2L` cache is built by:

- `buildM2LTables(const ssystem *sys)`

This function uses the flattened cube/pair/group metadata prepared in
[`src/fmm.c`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB/src/fmm.c).

It:

1. computes total storage for all M2L derivative tables
2. builds flattened coefficient offsets
3. computes `G0` and `Gk` for each M2L pair using `setupDerivs()`
4. uploads the derivative tables and indexing data to the GPU

The important point is that the expensive derivative tables are built once and
reused across all matvecs for the same geometry.

### Kernel

The grouped M2L kernel is:

- `m2lGroupedKernel`

This kernel launches one block per destination cube group.

For a target cube, it:

1. loops over all source cubes in its interaction list
2. reconstructs the same Cartesian-index convolution used by CPU `convM2L()`
3. accumulates into the four local expansion vectors:
   - `lec_k1`
   - `lec_k2`
   - `lec_k3`
   - `lec_k4`

This is the far-field analogue of the destination-grouped idea used in the
near-field kernel.

### Runtime entry point

The runtime call is:

- `gpuM2LApply(ssystem *sys)`

This function:

1. builds the M2L cache the first time a system is seen
2. packs current cube moments into flat arrays
3. zeros the GPU local-expansion buffers
4. launches the grouped M2L kernel
5. copies local expansions back to the host cubes

So the geometry-dependent part is cached, while the current moments are updated
every matvec.

## Leaf `Q2M` and `L2P`

### Host build

The leaf transform cache is built by:

- `buildLeafTables(const ssystem *sys)`

This function flattens the finest-level matrices:

- `Q2M0`
- `Q2M1`
- `L2P0`
- `L2P1`

These are precomputed on the CPU already by the original solver and are simply
repacked for GPU use.

### Kernels

The two leaf kernels are:

- `q2mLeafKernel`
- `l2pLeafKernel`

`q2mLeafKernel`:

- one block per leaf
- one thread per moment row
- applies the leaf matrix to the panel unknowns and writes leaf moments

`l2pLeafKernel`:

- one block per leaf
- threads walk destination panels within the leaf
- applies the local expansions back to panel values

### Runtime entry points

- `gpuQ2MApply(ssystem *sys, const double *sgm)`
- `gpuL2PApply(ssystem *sys, double alpha, double beta, double *pot)`

Both functions:

1. build the leaf cache once
2. move only the current vector data needed for the matvec
3. run the leaf kernel
4. copy results back into the existing CPU-side FMM structures

## Direct Dense GPU Baseline

The direct GPU baseline is built by:

- `buildDirectTables(const ssystem *sys)`

This creates the full dense panel-pair operator on the host using `panelIA0()`
for all source-target pairs.

The kernel is:

- `directApplyKernel`

It launches one block per destination panel and reduces the dense direct sum in
shared memory.

The runtime entry point is:

- `gpuDirectApply(ssystem *sys, double alpha, double beta, const double *sgm, double *pot)`

This path is intentionally memory-heavy and exists mainly to compare:

- direct GPU acceleration without FMM
- versus the GPU-FMM path

## CPU/GPU Split

The current implementation is hybrid, not fully GPU-resident.

Still on CPU:

- tree construction and traversal setup
- geometry-dependent quadrature such as `panelIA0()`
- some setup steps such as `setupRHS()`
- preconditioner application logic, except for cached setup reuse

Moved to GPU in the matvec:

- near-field application
- grouped `M2L`
- leaf `Q2M`
- leaf `L2P`

This is why the backend uses persistent caches: the expensive geometry and
translation tables are built once, while repeated GMRES matvec operations use
the GPU for the heavy apply steps.

## Main Idea

The CUDA backend accelerates the solver by combining:

- flattened coefficient storage
- grouped destination-based kernels
- reuse of geometry-dependent tables across repeated GMRES iterations

The most important design choice is that the GPU is used for repeated matvec
work, while the one-time geometry-specific quadrature remains on the CPU unless
it has already been repacked and cached.
