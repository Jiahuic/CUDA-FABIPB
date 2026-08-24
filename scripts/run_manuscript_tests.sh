#!/usr/bin/env bash
# Sequential driver for the manuscript_todo.md test items.
#
# Runs one test at a time -- never in parallel -- so each has the whole GPU.
# Every step records provenance (commit, load, GPU state, FABIPB_* environment)
# per results_plan.md section 0, and continues past a failing step.
#
# Usage:  ./scripts/run_manuscript_tests.sh [step ...]
#         with no arguments, runs the load-insensitive set.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="$root/results/paper/benchmarks"
bin="$root/build/fabipb"
runner="$root/scripts/run_6co8_fabipb_fast.sh"
mkdir -p "$bench"

# shellcheck source=scripts/lib_idle.sh
source "$root/scripts/lib_idle.sh"

log()  { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Provenance for one output directory.
provenance() {
  local d="$1"; mkdir -p "$d"
  git -C "$root" rev-parse HEAD           > "$d/git_commit.txt" 2>&1
  git -C "$root" status --short --branch  > "$d/git_status.txt" 2>&1
  cat /proc/loadavg                       > "$d/loadavg_before.txt"
  [[ "${IDLE_CONTENDED:-0}" == "1" ]] && \
    echo "host was contended at launch; times not usable for timing tables" > "$d/CONTENDED.txt"
  env | grep '^FABIPB_' | sort            > "$d/env_fabipb.txt" 2>/dev/null || true
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,memory.used \
             --format=csv                 > "$d/gpu_before.txt" 2>&1
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv                 > "$d/gpu_procs_before.txt" 2>&1
}
finish() {
  local d="$1"
  cat /proc/loadavg > "$d/loadavg_after.txt"
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu --format=csv > "$d/gpu_after.txt" 2>&1
}

# Run one command with the machine to ourselves, sampling contention throughout.
# $1=outdir  $2...=the command.  Sets RUN_RC.
guarded_run() {
  local d="$1"; shift
  local runpid
  "$@" &
  runpid=$!
  idle_watch_start "$d" "$runpid"
  wait "$runpid"; RUN_RC=$?
  idle_watch_stop "$d"
}

# Direct-binary step (no runner): $1=outdir name, rest=fabipb args
direct() {
  local name="$1"; shift
  local d="$bench/$name"
  if [[ -e "$d/fmm.log" ]]; then log "SKIP $name (exists)"; return 0; fi
  require_idle "$name" || { log "SKIPPED $name (machine busy)"; return 0; }
  provenance "$d"
  log "RUN  $name"
  guarded_run "$d" env -C "$root" FABIPB_REUSE_MESH=1 \
      /usr/bin/time -v -o "$d/time_v.txt" "$bin" "$@" >"$d/fmm.log" 2>&1
  finish "$d"
  printf '%s\n' "$*" > "$d/command.txt"
  log "DONE $name rc=$RUN_RC  $(grep -oE 'ttl time: [0-9.]+' "$d/fmm.log" | tail -1)$(
      [[ -e "$d/CONTENDED.txt" ]] && echo '  [CONTENDED]')"
  return 0
}

# Runner step: $1=outdir name, $2...=VAR=VAL env, last arg = pqr
runner_step() {
  local name="$1"; shift
  local pqr="${!#}"; set -- "${@:1:$(($#-1))}"
  local d="$bench/$name"
  if [[ -e "$d/fmm.log" ]]; then log "SKIP $name (exists)"; return 0; fi
  require_idle "$name" || { log "SKIPPED $name (machine busy)"; return 0; }
  provenance "$d"
  log "RUN  $name  ($*)"
  guarded_run "$d" env -C "$root" "$@" OUT_DIR="$d" ALLOW_EXISTING_OUT_DIR=1 \
      /usr/bin/time -v -o "$d/time_v.txt" "$runner" "$pqr" >"$d/driver.log" 2>&1
  finish "$d"
  log "DONE $name rc=$RUN_RC  $(grep -oE 'ttl time: [0-9.]+' "$d/fmm.log" 2>/dev/null | tail -1)$(
      [[ -e "$d/CONTENDED.txt" ]] && echo '  [CONTENDED]')"
  return 0
}


# The idle gate now lives in scripts/lib_idle.sh and runs before EVERY step
# rather than once per batch: a six-hour queue that checks the machine only at
# hour zero is not protected at hour five.  It also gates on the GPU, not just
# loadavg -- free VRAM decides the resident/streaming regime, so a foreign
# process changes which code path runs, not merely how fast it runs.

Q="-eps1=4 -eps2=80 -P=3 -q=1 -a=30 -i=100 -o=1e-4"
M7="$root/test_proteins/7A6A_charmm_protein_compact"
ZIKV="$root/test_proteins/ZIKV_6CO8_zenodo.pqr"
H1N1="$root/test_proteins/H1N1_atoms.pqr"

# ---- C2: near-field work assignment, resident regime (fast) ----------------
step_C2() {
  direct c2_workassign_dstleaf_7a6a -B=1 -g=1 -G=1 -m=2 -R=1.0 -t=6 $Q "$M7"
  direct c2_workassign_interaction_7a6a -B=1 -g=1 -G=0 -m=2 -R=1.0 -t=6 $Q "$M7"
}

# ---- C5a: streaming vs resident, same mesh (fast) --------------------------
step_C5a() {
  direct c5_resident_7a6a -B=1 -g=1 -G=1 -m=2 -R=1.0 -t=6 $Q "$M7"
  local d="$bench/c5_forced_streaming_7a6a"
  if [[ -e "$d/fmm.log" ]]; then log "SKIP c5_forced_streaming_7a6a (exists)"; return 0; fi
  provenance "$d"; log "RUN  c5_forced_streaming_7a6a"
  ( cd "$root" && FABIPB_REUSE_MESH=1 FABIPB_GPU_NEARFIELD_FORCE_STREAMING=1 \
      /usr/bin/time -v "$bin" -B=1 -g=1 -G=1 -m=2 -R=1.0 -t=6 $Q "$M7" ) \
      >"$d/fmm.log" 2>"$d/time_v.txt"
  finish "$d"; log "DONE c5_forced_streaming_7a6a  $(grep -oE 'ttl time: [0-9.]+' "$d/fmm.log" | tail -1)"
}

# ---- B2: does -q=3 disable the device disjoint path? (fast) ----------------
step_B2() {
  direct b2_qorder1_7a6a -B=1 -g=1 -m=2 -R=1.0 -t=6 -eps1=4 -eps2=80 -P=3 -q=1 -a=30 -i=100 -o=1e-4 "$M7"
  direct b2_qorder3_7a6a -B=1 -g=1 -m=2 -R=1.0 -t=6 -eps1=4 -eps2=80 -P=3 -q=3 -a=30 -i=100 -o=1e-4 "$M7"
}

# ---- C1: preconditioner residual history on the capsid ---------------------
step_C1() {
  runner_step zikv_sdens1_precond3 SDENS=1 FMM_PRECONDITIONER=3 \
      FABIPB_GMRES_LOG_RESID=1 "$ZIKV"
  runner_step zikv_sdens1_precond2 SDENS=1 FMM_PRECONDITIONER=2 \
      FABIPB_GMRES_LOG_RESID=1 GMRES_MAX_ITER=60 "$ZIKV"
}

# ---- C5b: auto Q2M/L2P policy on H1N1 sdens=0.5 ----------------------------
step_C5b() {
  runner_step h1n1_sdens05_q2m_forced_on SDENS=0.5 FMM_Q2M=1 "$H1N1"
}

# ---- C3: third ZIKV density for extrapolation ------------------------------
step_C3() {
  runner_step zikv_6co8_sdens15 SDENS=1.5 "$ZIKV"
}


# ---- C2b: depth sweep -- where does destination-leaf actually win? ---------
# C2 on 7A6A t=6 measured atomics FASTER (7.862 vs 9.973 ms/call), against the
# 68x claimed in the evidence base. Shallower trees mean fuller leaves and
# higher fan-in, which should favour destination ownership. This finds the
# crossover, or shows there isn't one.
step_C2b() {
  local d="$bench/c2b_depth_sweep"
  if [[ -e "$d/sweep.txt" ]]; then log "SKIP C2b (exists)"; return 0; fi
  provenance "$d"; log "RUN  C2b depth sweep"
  : > "$d/sweep.txt"
  for t in 4 5 6 7 8; do
    for G in 1 0; do
      local out k lv
      out=$( cd "$root" && FABIPB_REUSE_MESH=1 timeout 1800 "$bin" \
             -B=1 -g=1 -G=$G -m=2 -R=1.0 -t=$t $Q "$M7" 2>&1 )
      k=$(printf '%s' "$out" | grep -oE "kernel=[0-9.]+" | tail -1 | cut -d= -f2)
      lv=$(printf '%s' "$out" | grep -oE "leaf-cubes=[0-9]+" | cut -d= -f2)
      printf 't=%s G=%s leaves=%s kernel_ms_per_call=%s\n' "$t" "$G" "$lv" "$k" \
        | tee -a "$d/sweep.txt"
      printf '%s' "$out" > "$d/t${t}_G${G}.log"
    done
  done
  finish "$d"; log "DONE C2b"
}

# ---- A2: repeats (timing-critical; idle host required) ---------------------
step_A2() {
  local i
  for i in 1 2 3; do
    runner_step "zikv_6co8_sdens1_rep$i" SDENS=1 "$ZIKV"
  done
  for i in 1 2 3; do
    runner_step "h1n1_sdens05_rep_$i" SDENS=0.5 "$H1N1"
  done
}

# ---- A3: H1N1 sdens=1 into paper/benchmarks --------------------------------
step_A3() {
  local i
  for i in 1 2; do
    runner_step "h1n1_sdens1_rep$i" SDENS=1 FMM_DEPTH=9 "$H1N1"
  done
}

steps=("$@")
if [[ ${#steps[@]} -eq 0 ]]; then
  steps=(C2b C5a B2 C1 C5b C3 A2 A3)
fi
log "sequential run: ${steps[*]}   commit $(git -C "$root" rev-parse --short HEAD)"
log "idle gate: load<${IDLE_LOAD_THRESHOLD} foreign-cores<=${IDLE_FOREIGN_CORES} GPU<${IDLE_GPU_MEM_MIB}MiB, \
${IDLE_SUSTAIN_MIN}min sustained, checked before every step"
for s in "${steps[@]}"; do "step_$s"; done
log "all steps finished"

# Summarise which results are usable for timing, so a night's queue does not
# have to be audited by hand.
log "contention summary:"
for d in "$bench"/*/; do
  [[ -e "$d/contention_max.txt" ]] || continue
  printf '    %-40s %s %s\n' "$(basename "$d")" \
    "$([[ -e "$d/CONTENDED.txt" ]] && echo 'DIRTY ' || echo 'clean ')" \
    "$(cat "$d/contention_max.txt")"
done
