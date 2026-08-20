#!/usr/bin/env bash
# Profile the charge-tree energy kernel to decide whether it is bound by
# scratch traffic or by latency.
#
# The question this answers: ~70% of the per-thread derivative scratch is
# intermediate levels the contraction never reads (495 of 715 doubles at
# derivMax=8). Restructuring the recurrence to keep only two rows per level
# live would cut the footprint 2.8x, but that only pays if the kernel is
# actually memory-bound. These are the metrics that settle it:
#
#   dram__throughput / gpu__compute_memory_throughput  -- memory bound?
#   lts__t_sector_hit_rate                             -- is the scratch in L2?
#   l1tex__t_sector_hit_rate                           -- or in L1?
#   sm__throughput                                     -- or neither (latency)?
#
# Read it as: high DRAM throughput + low L2 hit rate => footprint reduction
# wins. Low DRAM throughput + low SM throughput => latency-bound, and the
# rewrite buys nothing.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mesh="${MESH:-$repo_root/test_proteins/7A6A_charmm_protein_compact}"
out="${OUT:-$repo_root/results/profile/energy_kernel_$(date -u +%Y%m%d_%H%M%S)}"
order="${ORDER:-8}"   # -p; 8 gives derivMax=9, matching the capsid
mkdir -p "$out"

metrics="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sector_hit_rate.pct,\
launch__occupancy_limit_registers,\
sm__warps_active.avg.pct_of_peak_sustained_active"

env FABIPB_REUSE_MESH=1 FABIPB_FORCE_TREE_RHS=1 \
    FABIPB_ENERGY_MODE=panel-tree FABIPB_ENERGY_GPU=1 \
    FABIPB_RHS_TREE_THETA=0.3 \
    FABIPB_RHS_THREADS=72 FABIPB_ENERGY_THREADS=72 \
    FABIPB_NEARFIELD_BUILD_THREADS=72 FABIPB_SETUP_THREADS=72 \
    FABIPB_PRECOND_APPLY_THREADS=72 FABIPB_DIRECT_THREADS=72 \
  ncu --target-processes all \
      --kernel-name chargeTreeEnergyKernel \
      --launch-count 1 \
      --metrics "$metrics" \
      --csv \
      "$repo_root/build/fabipb" -B=1 -g=1 -m=2 -R=1.0 -t=6 -p="$order" \
      -eps1=4 -eps2=80 -P=3 -q=1 -Q=1 -a=30 -i=100 -o=1e-4 \
      "$mesh" 2>&1 | tee "$out/ncu.csv"

echo "wrote $out/ncu.csv"
