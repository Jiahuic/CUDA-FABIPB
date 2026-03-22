#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
HYBRID_SETUP_THREADS="${HYBRID_SETUP_THREADS:-8}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/1ajj -t=5 -p=9" >&2
  exit 2
fi

panel="$1"
shift
case "$panel" in
  *.pqr) panel="${panel%.pqr}" ;;
esac

if [ ! -x "$BUILD_DIR/fabipb" ]; then
  echo "Error: $BUILD_DIR/fabipb not found. Build first:" >&2
  echo "  cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/direct_compare_logs/$timestamp}"
mkdir -p "$OUT_DIR"

cpu_fmm_log="$OUT_DIR/cpu_fmm.log"
cpu_direct_log="$OUT_DIR/cpu_direct.log"
gpu_direct_log="$OUT_DIR/gpu_direct.log"
hybrid_fmm_log="$OUT_DIR/hybrid_fmm.log"
prep_log="$OUT_DIR/prep.log"

mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 "$panel" "$@" >"$prep_log" 2>&1
fi

./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 -r=0 -m=0 "$panel" "$@" >"$cpu_fmm_log" 2>&1
./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 -r=2 -m=0 "$panel" "$@" >"$cpu_direct_log" 2>&1
./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -r=1 -m=0 "$panel" "$@" >"$gpu_direct_log" 2>&1
FABIPB_SETUP_THREADS="$HYBRID_SETUP_THREADS" \
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -Q=0 -r=0 -m=0 "$panel" "$@" >"$hybrid_fmm_log" 2>&1

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
    matvec_mode)
      awk -F'[= ,)]+' '/Matvec mode=/{val=$3} END{print val}' "$log"
      ;;
    near_ms)
      awk '/FMM stage avg\/call \(ms\):/{for(i=1;i<=NF;i++){if($i ~ /^Near=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
    direct_mem)
      awk '/Direct GPU memory estimate:/{sub(/^Direct GPU memory estimate: /,""); val=$0} END{print val}' "$log"
      ;;
    direct_fit)
      awk '/Direct GPU cache unavailable:/{val="no"} /GPU direct cache:/{val="yes"} END{if(val=="") val="unknown"; print val}' "$log"
      ;;
    direct_mode)
      awk '/GPU direct cache:/{for(i=1;i<=NF;i++){if($i ~ /^mode=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
  esac
}

cpu_fmm_ttl="$(extract_metric "$cpu_fmm_log" ttl)"
cpu_fmm_its="$(extract_metric "$cpu_fmm_log" its)"
cpu_fmm_energy="$(extract_metric "$cpu_fmm_log" energy)"
cpu_fmm_near_ms="$(extract_metric "$cpu_fmm_log" near_ms)"

cpu_direct_ttl="$(extract_metric "$cpu_direct_log" ttl)"
cpu_direct_its="$(extract_metric "$cpu_direct_log" its)"
cpu_direct_energy="$(extract_metric "$cpu_direct_log" energy)"

gpu_direct_ttl="$(extract_metric "$gpu_direct_log" ttl)"
gpu_direct_its="$(extract_metric "$gpu_direct_log" its)"
gpu_direct_energy="$(extract_metric "$gpu_direct_log" energy)"
gpu_direct_mem="$(extract_metric "$gpu_direct_log" direct_mem)"
gpu_direct_fit="$(extract_metric "$gpu_direct_log" direct_fit)"
gpu_direct_mode="$(extract_metric "$gpu_direct_log" direct_mode)"
gpu_direct_matvec_mode="$(extract_metric "$gpu_direct_log" matvec_mode)"

hybrid_fmm_ttl="$(extract_metric "$hybrid_fmm_log" ttl)"
hybrid_fmm_its="$(extract_metric "$hybrid_fmm_log" its)"
hybrid_fmm_energy="$(extract_metric "$hybrid_fmm_log" energy)"
hybrid_fmm_near_ms="$(extract_metric "$hybrid_fmm_log" near_ms)"

cpu_direct_vs_gpu_direct="$(awk -v c="$cpu_direct_ttl" -v g="$gpu_direct_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", c/g}}')"
cpu_fmm_vs_hybrid_fmm="$(awk -v c="$cpu_fmm_ttl" -v g="$hybrid_fmm_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", c/g}}')"
gpu_direct_vs_hybrid_fmm="$(awk -v d="$gpu_direct_ttl" -v g="$hybrid_fmm_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", d/g}}')"
cpu_direct_vs_hybrid_fmm="$(awk -v c="$cpu_direct_ttl" -v g="$hybrid_fmm_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", c/g}}')"

abs_direct_delta="$(awk -v a="$cpu_direct_energy" -v b="$gpu_direct_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"
abs_fmm_delta="$(awk -v a="$cpu_fmm_energy" -v b="$hybrid_fmm_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"

echo "Direct/FMM comparison run"
echo "  panel: $panel"
if [ -f "$prep_log" ]; then
  echo "  prep:  $prep_log"
fi
echo "  cpu-fmm:    ttl=${cpu_fmm_ttl}s its=${cpu_fmm_its} energy=${cpu_fmm_energy} near_ms=${cpu_fmm_near_ms}"
echo "  cpu-direct: ttl=${cpu_direct_ttl}s its=${cpu_direct_its} energy=${cpu_direct_energy}"
echo "  gpu-direct: ttl=${gpu_direct_ttl}s its=${gpu_direct_its} energy=${gpu_direct_energy} fit=${gpu_direct_fit}"
if [ -n "${gpu_direct_mode:-}" ]; then
  echo "              cache-mode: ${gpu_direct_mode}"
fi
if [ -n "${gpu_direct_mem:-}" ]; then
  echo "              memory: ${gpu_direct_mem}"
fi
if [ "${gpu_direct_fit}" != "yes" ] || [ "${gpu_direct_matvec_mode}" != "1" ]; then
  echo "              note: direct GPU did not stay on the requested direct path"
fi
echo "  hybrid-fmm: ttl=${hybrid_fmm_ttl}s its=${hybrid_fmm_its} energy=${hybrid_fmm_energy} near_ms=${hybrid_fmm_near_ms}"
echo "  delta:      |cpu-direct-gpu-direct|=${abs_direct_delta} |cpu-fmm-hybrid-fmm|=${abs_fmm_delta}"
echo "  speedup:    cpu-direct/gpu-direct=${cpu_direct_vs_gpu_direct}x cpu-fmm/hybrid-fmm=${cpu_fmm_vs_hybrid_fmm}x gpu-direct/hybrid-fmm=${gpu_direct_vs_hybrid_fmm}x cpu-direct/hybrid-fmm=${cpu_direct_vs_hybrid_fmm}x"
echo
echo "CPU FMM log:    $cpu_fmm_log"
echo "CPU direct log: $cpu_direct_log"
echo "GPU direct log: $gpu_direct_log"
echo "Hybrid FMM log: $hybrid_fmm_log"
