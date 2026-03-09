#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build-cuda}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/1a7m -t=5 -p=9" >&2
  exit 2
fi

panel="$1"
shift
case "$panel" in
  *.pqr) panel="${panel%.pqr}" ;;
esac

if [ ! -x "$BUILD_DIR/coulomb" ]; then
  echo "Error: $BUILD_DIR/coulomb not found. Build first:" >&2
  echo "  cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/compare_logs/$timestamp}"
mkdir -p "$OUT_DIR"

cpu_log="$OUT_DIR/cpu.log"
gpu_log="$OUT_DIR/gpu.log"
prep_log="$OUT_DIR/prep.log"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  "$BUILD_DIR/coulomb" -g=0 "$panel" "$@" >"$prep_log" 2>&1
fi

"$BUILD_DIR/coulomb" -g=0 -m=0 "$panel" "$@" >"$cpu_log" 2>&1
"$BUILD_DIR/coulomb" -g=1 -m=0 "$panel" "$@" >"$gpu_log" 2>&1

extract_metric() {
  log="$1"
  key="$2"
  case "$key" in
    ttl)
      awk '/ttl time:/{gsub(",", "", $3); val=$3} END{print val}' "$log"
      ;;
    its)
      awk '/ttl time:/{split($4,a,"="); val=a[2]} END{print val}' "$log"
      ;;
    energy)
      awk '/solvation energy:/{val=$3} END{print val}' "$log"
      ;;
    calls)
      awk '/FMM matvec stats:/{for(i=1;i<=NF;i++){if($i ~ /^calls=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
    near_ms)
      awk '/FMM stage avg\/call \(ms\):/{for(i=1;i<=NF;i++){if($i ~ /^Near=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
  esac
}

cpu_ttl="$(extract_metric "$cpu_log" ttl)"
gpu_ttl="$(extract_metric "$gpu_log" ttl)"
cpu_its="$(extract_metric "$cpu_log" its)"
gpu_its="$(extract_metric "$gpu_log" its)"
cpu_energy="$(extract_metric "$cpu_log" energy)"
gpu_energy="$(extract_metric "$gpu_log" energy)"
cpu_calls="$(extract_metric "$cpu_log" calls)"
gpu_calls="$(extract_metric "$gpu_log" calls)"
cpu_near_ms="$(extract_metric "$cpu_log" near_ms)"
gpu_near_ms="$(extract_metric "$gpu_log" near_ms)"

abs_energy_delta="$(awk -v a="$cpu_energy" -v b="$gpu_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"
speedup="$(awk -v c="$cpu_ttl" -v g="$gpu_ttl" 'BEGIN{if(g==0){print "inf"}else{printf "%.6f", c/g}}')"

echo "Comparison run"
echo "  panel: $panel"
if [ -f "$prep_log" ]; then
  echo "  prep:  $prep_log"
fi
echo "  cpu:   ttl=${cpu_ttl}s its=${cpu_its} energy=${cpu_energy} calls=${cpu_calls} near_ms=${cpu_near_ms}"
echo "  gpu:   ttl=${gpu_ttl}s its=${gpu_its} energy=${gpu_energy} calls=${gpu_calls} near_ms=${gpu_near_ms}"
echo "  delta: |energy_cpu-energy_gpu|=${abs_energy_delta} speedup(cpu/gpu)=${speedup}x"
echo
echo "CPU log: $cpu_log"
echo "GPU log: $gpu_log"
