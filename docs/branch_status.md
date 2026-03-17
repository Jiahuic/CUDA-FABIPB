# Branch Status

This note records which local branches have already been merged into
[`main`](/home/jiahuic/Garage/electrostatics/GPU-FABIPB) and which branches
should remain available for future work.

## Merged Into `main`

These branches are already present in `main` history and are safe to delete as
local working branches:

- `m2l-destination-grouped-experiment`
- `m2l-setup-opt-experiment`
- `nearfield-cache-persist`
- `nearfield-roadmap`
- `panelia0-parallel-experiment`
- `preconditioner-cache-experiment`
- `preconditioner-lu-experiment`
- `q2m-l2p-gpu-experiment`
- `rhs-gpu-experiment`
- `setup-parallel-experiment`

Notes:

- `nearfield-cache-persist` was an experiment that we decided not to carry
  forward, but the branch is still merged relative to `main` and can be removed
  as a local branch.
- `nearfield-roadmap` is historical/documentation work and does not need to
  remain as a separate local branch.

## Active Unmerged Branches To Keep

These should remain available for continuing work:

- `adaptive-nlev-directsum-plan`
  - adaptive `nLev`
  - direct-sum comparison and blocked direct GPU work
  - structural/order sweep scripts
- `nbody-fmm-prototype`
  - standalone particle FMM prototype under `src_nbody/`
- `nbody-recovery-plan`
  - reference/recovery notes for the older general N-body code

## Unmerged Branches Needing Explicit Decision

These are not merged into `main`, but they are also not part of the current
active plan. They should be reviewed before deletion:

- `nearfield-grouped-experiment`

Recommended handling:

1. Delete merged branches after recording them here.
2. Keep active unmerged branches until the related work is finished or merged.
3. Review stale unmerged branches individually instead of deleting them by
   default.
