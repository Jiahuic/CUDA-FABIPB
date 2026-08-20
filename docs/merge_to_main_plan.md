# Merge To Main Plan

Date: 2026-08-20

Branch:

```text
issue-zenodo-pqr-zero-radius
```

Status at plan time:

```text
main ahead of branch: 0 commits
branch ahead of main: 41 commits
```

## Goal

Merge the capsid GPU work into `main` after committing the latest working-tree
changes:

- GPU nearfield memory arbitration
- GPU nearfield stream-pair and special-cache support
- GPU M2L diagnostics and safer overflow handling
- GPU final energy reduction and timing breakdown
- source-owned defaults for capsid runs:
  - GPU auto-detect
  - Q2M/L2P auto placement
  - preconditioner auto selection
  - RHS and energy tree theta default `0.3`
  - panel-tree energy default
- post-solve energy tree theta default changed to `0.3`
- viral capsid performance documentation

## Stage Only These Files

```sh
git add README.md include/gk.h include/gpu_backend.h src/fabipb.c src/fmm.c src/gpu_backend_cuda.cu src/gpu_backend_stub.c src/treecode.c scripts/run_6co8_fabipb_fast.sh docs/6co8_gpu_fabipb_milestone.md docs/setup_rhs_and_gpu_tracking.md docs/viral_capsid_gpu_performance_2026_08_20.md docs/merge_to_main_plan.md
```

## Do Not Stage

The following are local datasets, builds, benchmark outputs, or external source
trees and should stay out of the merge unless intentionally curated later:

```text
TABI-PB/
nanoshaper-master
nanoshaper/
results/
resume.md
test_proteins/*.pdb
test_proteins/*_charmm.pqr
test_proteins/*_charmm_protein_compact.pqr
test_proteins/TABI.pqr
```

## Commit

```sh
git commit -m "perf: auto-tune capsid gpu defaults"
```

## Verify On Feature Branch

```sh
cmake --build build -j 8
bash -n scripts/run_6co8_fabipb_fast.sh
git diff --check
```

Local validation already run on this branch:

```text
cmake --build build -j 8                         passed
bash -n scripts/run_6co8_fabipb_fast.sh          passed
git diff --check                                  passed
STOP_AFTER_RHS=1 1a63 source-default smoke        passed
```

## Merge

```sh
git switch main
git merge --no-ff issue-zenodo-pqr-zero-radius
```

## Verify On Main

```sh
cmake --build build -j 8
git status --short
git log --oneline --max-count=5
```

## Notes

The main merge risk is accidentally staging large benchmark/data folders. Keep
the merge focused on source, runner, and documentation changes.
