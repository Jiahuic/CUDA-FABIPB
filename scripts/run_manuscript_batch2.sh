#!/usr/bin/env bash
# Second manuscript test batch: T6, T2, T4-test, T7, T3.
# Sequential -- one test at a time, whole GPU each. Resumable: a step whose
# output already exists is skipped.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bench="$root/results/paper/benchmarks"
bin="$root/build/fabipb"
runner="$root/scripts/run_6co8_fabipb_fast.sh"
ZIKV="$root/test_proteins/ZIKV_6CO8_zenodo.pqr"
H1N1="$root/test_proteins/H1N1_atoms.pqr"
mkdir -p "$bench"

# shellcheck source=scripts/lib_idle.sh
source "$root/scripts/lib_idle.sh"

log() { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

provenance() {
  local d="$1"; mkdir -p "$d"
  git -C "$root" rev-parse HEAD > "$d/git_commit.txt" 2>&1
  cat /proc/loadavg > "$d/loadavg_before.txt"
  env | grep '^FABIPB_' | sort > "$d/env_fabipb.txt" 2>/dev/null || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv > "$d/gpu_procs_before.txt" 2>&1
  [[ "${IDLE_CONTENDED:-0}" == "1" ]] && \
    echo "host was contended at launch; times not usable" > "$d/CONTENDED.txt"
}

guarded_run() {
  local d="$1"; shift
  local runpid
  "$@" &
  runpid=$!
  idle_watch_start "$d" "$runpid"
  wait "$runpid"; RUN_RC=$?
  idle_watch_stop "$d"
}

runner_step() {
  local name="$1"; shift
  local pqr="${!#}"; set -- "${@:1:$(($#-1))}"
  local d="$bench/$name"
  [[ -e "$d/fmm.log" ]] && { log "SKIP $name"; return 0; }
  require_idle "$name" || { log "SKIPPED $name (machine busy)"; return 0; }
  provenance "$d"; log "RUN  $name ($*)"
  guarded_run "$d" env -C "$root" "$@" OUT_DIR="$d" ALLOW_EXISTING_OUT_DIR=1 \
      /usr/bin/time -v -o "$d/time_v.txt" "$runner" "$pqr" >"$d/driver.log" 2>&1
  cat /proc/loadavg > "$d/loadavg_after.txt"
  log "DONE $name rc=$RUN_RC  $(grep -oE 'ttl time: [0-9.]+' "$d/fmm.log" 2>/dev/null | tail -1)$(
      [[ -e "$d/CONTENDED.txt" ]] && echo '  [CONTENDED]')"
}

# ---- T6: does the depth-5 work-assignment win hold on another geometry? ----
step_T6() {
  local mesh="$1"
  local tag="$2"
  local d="$bench/t6_workassign_$tag"
  [[ -e "$d/sweep.txt" ]] && { log "SKIP T6 $tag"; return 0; }
  # Reports per-call kernel time, so it is as timing-sensitive as any run.
  require_idle "T6 $tag" || { log "SKIPPED T6 $tag (machine busy)"; return 0; }
  provenance "$d"; log "RUN  T6 work-assignment sweep on $tag"
  : > "$d/sweep.txt"
  for t in 4 5 6 7; do
    for G in 1 0; do
      local out k lv
      out=$( cd "$root" && FABIPB_REUSE_MESH=1 timeout 1800 "$bin" \
             -B=1 -g=1 -G=$G -m=2 -R=1.0 -t=$t -eps1=4 -eps2=80 -P=3 -q=1 \
             -a=30 -i=100 -o=1e-4 "$mesh" 2>&1 )
      k=$(printf '%s' "$out" | grep -oE "kernel=[0-9.]+" | tail -1 | cut -d= -f2)
      lv=$(printf '%s' "$out" | grep -oE "leaf-cubes=[0-9]+" | cut -d= -f2)
      printf 't=%s G=%s leaves=%s kernel_ms_per_call=%s\n' "$t" "$G" "$lv" "$k" \
        | tee -a "$d/sweep.txt"
    done
  done
  log "DONE T6 $tag"
}

# ---- T2: depth study at capsid scale ---------------------------------------
step_T2() {
  runner_step h1n1_sdens05_depth9  SDENS=0.5 FMM_DEPTH=9  "$H1N1"
  runner_step h1n1_sdens05_depth10 SDENS=0.5 FMM_DEPTH=10 "$H1N1"
  runner_step zikv_6co8_sdens1_depth9 SDENS=1 FMM_DEPTH=9 "$ZIKV"
}

# ---- T4 test: halved leaf cache, can GPU Q2M/L2P stay on at capsid scale? --
step_T4test() {
  runner_step h1n1_sdens05_q2m_on_halvedcache SDENS=0.5 FMM_Q2M=1 "$H1N1"
}

# ---- T7: clean sdens=2 for the convergence series --------------------------
step_T7() {
  runner_step zikv_6co8_sdens2 SDENS=2 FMM_DEPTH=9 "$ZIKV"
}

# ---- T3: serial CPU reference (longest; last) ------------------------------
step_T3() {
  runner_step zikv_6co8_sdens1_cpu_serial \
      SDENS=1 FMM_GPU=0 FABIPB_NEARFIELD_APPLY_THREADS=1 "$ZIKV"
}

steps=("$@")
[[ ${#steps[@]} -eq 0 ]] && steps=(T6 T2 T4test T7 T3)
log "batch2: ${steps[*]}  commit $(git -C "$root" rev-parse --short HEAD)  load=$(awk '{print $3}' /proc/loadavg)"
for s in "${steps[@]}"; do
  case "$s" in
    T6) step_T6 "$root/test_proteins/1a63" 1a63 ;;
    *)  "step_$s" ;;
  esac
done
log "batch2 finished"
