# Shared machine-idle gate for timing runs.  Source, don't execute.
#
# A benchmark on a shared machine is only a measurement if nothing else is
# competing for the two resources we use.  Both matter, for different reasons:
#
#   CPU   -- mesh generation, the RHS charge tree, the host nearfield apply and
#            the GMRES Arnoldi are all threaded across every core.  A loaded
#            host inflates wall clock without touching any GPU number.
#   GPU   -- beyond kernel contention, free VRAM decides the near/far regime.
#            gpuNearfieldApply picks resident vs streaming from cudaMemGetInfo,
#            so a foreign process holding memory can change which code path
#            runs.  That is not a slow measurement, it is a different one.
#
# Usage:
#   source "$(dirname "$0")/lib_idle.sh"
#   require_idle "$name"       # blocks until idle, or refuses the step
#   idle_watch_start "$outdir" # sample contention during the run
#   idle_watch_stop  "$outdir" # writes contention_max.txt, flags CONTENDED
#
# Tunables (env):
#   IDLE_LOAD_THRESHOLD   15-min loadavg ceiling            (default: cores/8)
#   IDLE_GPU_MEM_MIB      foreign VRAM ceiling, MiB         (default 512)
#   IDLE_GPU_UTIL         foreign utilization ceiling, %    (default 10)
#   IDLE_SUSTAIN_MIN      consecutive clean minutes needed  (default 3)
#   IDLE_MAX_WAIT_MIN     give up after this many minutes   (default 420)
#   ALLOW_CONTENDED=1     on give-up, run anyway and mark   (default: refuse)
#   SKIP_IDLE_WAIT=1      bypass the gate entirely
#
# Default is to REFUSE a contended step rather than burn GPU hours producing a
# number the timing tables will discard.  ALLOW_CONTENDED=1 restores the old
# run-anyway behaviour for cases where iterations/energy are the point and the
# clock is not.

IDLE_CORES="$(nproc)"
: "${IDLE_LOAD_THRESHOLD:=$((IDLE_CORES / 8))}"
: "${IDLE_GPU_MEM_MIB:=512}"
: "${IDLE_GPU_UTIL:=10}"
: "${IDLE_SUSTAIN_MIN:=3}"
: "${IDLE_MAX_WAIT_MIN:=420}"

: "${IDLE_SAMPLE_SEC:=60}"
# Foreign cores tolerated mid-run before a run is marked contended.  Small but
# nonzero: sshd, the sampler itself and cron noise are not interference.
: "${IDLE_FOREIGN_CORES:=2}"

idle_log() { printf '    [idle %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# ---- foreign CPU load ------------------------------------------------------
# During a run the 15-minute loadavg is useless as a contention signal: our own
# solver threads every core by design and would flag every legitimate run.  We
# need load that is NOT ours.  Measure total busy cores from /proc/stat and
# subtract the cores consumed by our own process tree, both as deltas over the
# same short interval, which also makes the reading instantaneous rather than a
# 15-minute tail.

_idle_stat_busy_jiffies() {  # total non-idle jiffies across all CPUs
  awk '/^cpu /{ idle=$5+$6; tot=0; for(i=2;i<=NF;i++) tot+=$i; print tot-idle, tot; exit }' /proc/stat
}

# utime+stime for pid $1 and every descendant, in jiffies.
#
# Walks the PPID tree rather than matching a process group: a child started as
# `( cmd & )` -- which the runner does -- lands in its own process group, so
# group matching misses exactly the workers we need to exclude.  Parsing skips
# past the last ')' because a process name may contain spaces or parentheses.
_idle_tree_jiffies() {
  local root="${1:-}"
  [[ -z "$root" ]] && { echo 0; return; }
  awk -v root="$root" '
    BEGIN {
      while (("ls -1 /proc 2>/dev/null" | getline d) > 0) {
        if (d !~ /^[0-9]+$/) continue
        f = "/proc/" d "/stat"
        if ((getline line < f) > 0) {
          p = index(line, ") ")
          if (p == 0) { close(f); continue }
          rest = substr(line, p + 2)
          n = split(rest, a, " ")
          if (n < 13) { close(f); continue }
          ppid[d] = a[2]; jif[d] = a[12] + a[13]; pids[++np] = d
        }
        close(f)
      }
      # Mark root and everything descended from it.  Repeat until stable so
      # parents appearing after children in readdir order still propagate.
      mine[root] = 1
      do {
        changed = 0
        for (i = 1; i <= np; i++) {
          d = pids[i]
          if (!mine[d] && (d in ppid) && mine[ppid[d]]) { mine[d] = 1; changed = 1 }
        }
      } while (changed)
      total = 0
      for (i = 1; i <= np; i++) if (mine[pids[i]]) total += jif[pids[i]]
      print total
    }'
}

# Cores busy with work that is not ours.  $1 = our process-group leader (opt).
_idle_foreign_cores() {
  local own="${1:-}" hz b0 t0 o0 b1 t1 o1 dt
  hz=$(getconf CLK_TCK 2>/dev/null || echo 100)
  read -r b0 t0 < <(_idle_stat_busy_jiffies); o0=$(_idle_tree_jiffies "$own")
  sleep 1
  read -r b1 t1 < <(_idle_stat_busy_jiffies); o1=$(_idle_tree_jiffies "$own")
  dt=$(( t1 - t0 ))
  (( dt <= 0 )) && { echo 0; return; }
  # (busy_delta - own_delta) / elapsed_jiffies_per_cpu * ncpu == foreign cores
  awk -v b="$(( b1 - b0 ))" -v o="$(( o1 - o0 ))" -v d="$dt" -v n="$IDLE_CORES" \
      'BEGIN{ f=(b-o)/d*n; if(f<0) f=0; printf "%.2f", f }'
}

# PIDs of our own fabipb processes, so the sampler does not flag us as foreign.
_idle_own_pids() { pgrep -u "$(id -u)" -x fabipb 2>/dev/null | tr '\n' ' '; }

# Foreign GPU memory in MiB: total used, minus anything our own fabipb holds.
# When per-process accounting returns nothing -- processes in other containers,
# or MIG -- this counts all used memory as foreign.  That is the conservative
# direction: we would rather wait than mismeasure.
_idle_gpu_foreign_mib() {
  local used own_mib=0 pid mib own_pids
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ -z "$used" ]] && { echo 0; return; }
  own_pids=" $(_idle_own_pids) "
  while IFS=, read -r pid mib; do
    pid="${pid// /}"; mib="${mib// /}"
    [[ -z "$pid" ]] && continue
    [[ "$own_pids" == *" $pid "* ]] && own_mib=$((own_mib + ${mib:-0}))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory \
             --format=csv,noheader,nounits 2>/dev/null)
  echo $(( used - own_mib ))
}

# Count of compute processes that are not ours.
_idle_gpu_foreign_procs() {
  local pid n=0 own_pids
  own_pids=" $(_idle_own_pids) "
  while read -r pid; do
    pid="${pid// /}"
    [[ -z "$pid" ]] && continue
    [[ "$own_pids" == *" $pid "* ]] || n=$((n + 1))
  done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
  echo "$n"
}

# Echoes a human reason if the machine is busy, nothing if it is clean.
idle_reason() {
  local l15 util procs mib
  l15=$(awk '{print $3}' /proc/loadavg)
  procs=$(_idle_gpu_foreign_procs)
  mib=$(_idle_gpu_foreign_mib)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)

  if awk -v a="$l15" -v b="$IDLE_LOAD_THRESHOLD" 'BEGIN{exit !(a>=b)}'; then
    echo "host load15=$l15 >= $IDLE_LOAD_THRESHOLD"; return
  fi
  if (( procs > 0 )); then
    echo "$procs foreign GPU compute process(es)"; return
  fi
  if (( mib > IDLE_GPU_MEM_MIB )); then
    echo "foreign GPU memory ${mib} MiB > ${IDLE_GPU_MEM_MIB} MiB"; return
  fi
  # Utilization is only meaningful once no foreign process holds memory; our
  # own run legitimately pegs it at 100%.
  if (( ${util:-0} > IDLE_GPU_UTIL )); then
    echo "GPU utilization ${util}% > ${IDLE_GPU_UTIL}% with no attributable process"; return
  fi
  echo ""
}

# Block until the machine is idle.  Returns 0 to proceed, 1 to skip the step.
# Sets IDLE_CONTENDED=1 when proceeding under ALLOW_CONTENDED.
require_idle() {
  local what="${1:-step}" ok=0 waited=0 why
  IDLE_CONTENDED=0
  if [[ "${SKIP_IDLE_WAIT:-0}" == "1" ]]; then
    idle_log "gate bypassed for $what"; return 0
  fi

  why="$(idle_reason)"
  if [[ -z "$why" ]]; then
    idle_log "machine clean; need ${IDLE_SUSTAIN_MIN} sustained min before $what"
  else
    idle_log "waiting for $what: $why"
  fi

  while (( waited < IDLE_MAX_WAIT_MIN )); do
    why="$(idle_reason)"
    if [[ -z "$why" ]]; then ok=$((ok + 1)); else ok=0; fi
    if (( ok >= IDLE_SUSTAIN_MIN )); then
      idle_log "clean on ${ok} consecutive checks over ${waited} min (load15=$(awk '{print $3}' /proc/loadavg), GPU free) -- starting $what"
      return 0
    fi
    sleep 60; waited=$((waited + 1))
    if (( waited % 15 == 0 )); then
      idle_log "still waiting (${waited}/${IDLE_MAX_WAIT_MIN} min): $why"
    fi
  done

  if [[ "${ALLOW_CONTENDED:-0}" == "1" ]]; then
    idle_log "gave up after ${IDLE_MAX_WAIT_MIN} min; running $what ANYWAY, marked contended"
    IDLE_CONTENDED=1
    return 0
  fi
  idle_log "gave up after ${IDLE_MAX_WAIT_MIN} min ($why). SKIPPING $what."
  idle_log "set ALLOW_CONTENDED=1 to run regardless (times will be excluded)."
  return 1
}

# ---- contention sampling during a run --------------------------------------
# A gate that only fires before launch says nothing about hour six of a
# six-hour run.  Sample throughout and keep the worst reading, so the tables
# judge a run on what the machine did while it ran, not on one snapshot.


# $1 = output dir, $2 = pid of our own run, whose whole descendant tree is
# excluded from the foreign-CPU figure (our solver saturates every core by
# design, so total load cannot distinguish us from interference).
# Columns: epoch load15 foreign_cores gpu_procs gpu_mib
idle_watch_start() {
  local d="$1" own="${2:-}"
  # Take one sample synchronously before backgrounding the loop.  A run shorter
  # than the sample interval would otherwise finish with zero samples, and an
  # empty sample set must never be mistaken for a clean one.
  { echo "# epoch load15 foreign_cores foreign_gpu_procs foreign_gpu_mib"
    printf '%s %s %s %s %s\n' "$(date +%s)" \
      "$(awk '{print $3}' /proc/loadavg)" "$(_idle_foreign_cores "$own")" \
      "$(_idle_gpu_foreign_procs)" "$(_idle_gpu_foreign_mib)"
  } > "$d/contention_samples.txt"
  ( while :; do
      sleep "$IDLE_SAMPLE_SEC"
      printf '%s %s %s %s %s\n' "$(date +%s)" \
        "$(awk '{print $3}' /proc/loadavg)" \
        "$(_idle_foreign_cores "$own")" \
        "$(_idle_gpu_foreign_procs)" \
        "$(_idle_gpu_foreign_mib)"
      sleep "$IDLE_SAMPLE_SEC"
    done ) > "$d/contention_samples.txt" 2>/dev/null &
  IDLE_WATCH_PID=$!
}

idle_watch_stop() {
  local d="$1"
  if [[ -n "${IDLE_WATCH_PID:-}" ]]; then
    kill "$IDLE_WATCH_PID" 2>/dev/null
    wait "$IDLE_WATCH_PID" 2>/dev/null
    IDLE_WATCH_PID=""
  fi
  [[ -s "$d/contention_samples.txt" ]] || return 0
  # Dirty is judged on FOREIGN cores, not loadavg: our own solver saturates
  # every core by design, so total load says nothing about interference.
  awk -v cap="$IDLE_FOREIGN_CORES" -v memcap="$IDLE_GPU_MEM_MIB" '
    /^#/ { next }
    { n++
      if ($2+0 > maxl) maxl = $2+0
      if ($3+0 > maxf) maxf = $3+0
      if ($4+0 > maxp) maxp = $4+0
      if ($5+0 > maxm) maxm = $5+0
      if ($3+0 > cap || $4+0 > 0 || $5+0 > memcap) dirty++ }
    END {
      # No samples means no evidence, which is NOT evidence of a quiet machine.
      # Report it as unknown so nothing downstream can read it as clean.
      d = (n == 0) ? "unknown" : dirty+0
      printf "samples=%d max_load15=%.2f max_foreign_cores=%.2f max_foreign_gpu_procs=%d max_foreign_gpu_mib=%d dirty_samples=%s foreign_core_cap=%s\n",
             n, maxl, maxf, maxp, maxm, d, cap }
  ' "$d/contention_samples.txt" > "$d/contention_max.txt"
  if ! grep -q 'dirty_samples=0 ' "$d/contention_max.txt"; then
    echo "contention observed DURING the run, or no samples taken; see contention_max.txt" \
      > "$d/CONTENDED.txt"
  fi
}
