#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"

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

if [ ! -x "$BUILD_DIR/coulomb" ]; then
  echo "Error: $BUILD_DIR/coulomb not found. Build first:" >&2
  echo "  cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/direct_compare_logs/$timestamp}"
mkdir -p "$OUT_DIR"

cpu_log="$OUT_DIR/cpu.log"
direct_log="$OUT_DIR/direct_gpu.log"
fmm_gpu_log="$OUT_DIR/fmm_gpu.log"
prep_log="$OUT_DIR/prep.log"

mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/coulomb" -g=0 "$panel" "$@" >"$prep_log" 2>&1
fi

./scripts/with_benchmark_env.sh "$BUILD_DIR/coulomb" -g=0 -m=0 "$panel" "$@" >"$cpu_log" 2>&1
./scripts/with_benchmark_env.sh "$BUILD_DIR/coulomb" -g=1 -r=1 -m=0 "$panel" "$@" >"$direct_log" 2>&1
./scripts/with_benchmark_env.sh "$BUILD_DIR/coulomb" -g=1 -m=0 "$panel" "$@" >"$fmm_gpu_log" 2>&1

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
    near_ms)
      awk '/FMM stage avg\/call \(ms\):/{for(i=1;i<=NF;i++){if($i ~ /^Near=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
    direct_mem)
      awk '/Direct GPU memory estimate:/{sub(/^Direct GPU memory estimate: /,""); val=$0} END{print val}' "$log"
      ;;
    direct_fit)
      awk '/Direct GPU cache unavailable:/{val="no"} /GPU direct cache:/{val="yes"} END{if(val=="") val="unknown"; print val}' "$log"
      ;;
  esac
}

cpu_ttl="$(extract_metric "$cpu_log" ttl)"
cpu_its="$(extract_metric "$cpu_log" its)"
cpu_energy="$(extract_metric "$cpu_log" energy)"
cpu_near_ms="$(extract_metric "$cpu_log" near_ms)"

direct_ttl="$(extract_metric "$direct_log" ttl)"
direct_its="$(extract_metric "$direct_log" its)"
direct_energy="$(extract_metric "$direct_log" energy)"
direct_mem="$(extract_metric "$direct_log" direct_mem)"
direct_fit="$(extract_metric "$direct_log" direct_fit)"

fmm_gpu_ttl="$(extract_metric "$fmm_gpu_log" ttl)"
fmm_gpu_its="$(extract_metric "$fmm_gpu_log" its)"
fmm_gpu_energy="$(extract_metric "$fmm_gpu_log" energy)"
fmm_gpu_near_ms="$(extract_metric "$fmm_gpu_log" near_ms)"

cpu_vs_direct="$(awk -v c="$cpu_ttl" -v g="$direct_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", c/g}}')"
cpu_vs_fmm_gpu="$(awk -v c="$cpu_ttl" -v g="$fmm_gpu_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", c/g}}')"
direct_vs_fmm_gpu="$(awk -v d="$direct_ttl" -v g="$fmm_gpu_ttl" 'BEGIN{if(g==0||g==""){print "inf"}else{printf "%.6f", d/g}}')"

abs_direct_delta="$(awk -v a="$cpu_energy" -v b="$direct_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"
abs_fmm_delta="$(awk -v a="$cpu_energy" -v b="$fmm_gpu_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"

echo "Direct GPU comparison run"
echo "  panel: $panel"
if [ -f "$prep_log" ]; then
  echo "  prep:  $prep_log"
fi
echo "  cpu:        ttl=${cpu_ttl}s its=${cpu_its} energy=${cpu_energy} near_ms=${cpu_near_ms}"
echo "  direct-gpu: ttl=${direct_ttl}s its=${direct_its} energy=${direct_energy} fit=${direct_fit}"
if [ -n "${direct_mem:-}" ]; then
  echo "              memory: ${direct_mem}"
fi
echo "  fmm-gpu:    ttl=${fmm_gpu_ttl}s its=${fmm_gpu_its} energy=${fmm_gpu_energy} near_ms=${fmm_gpu_near_ms}"
echo "  delta:      |cpu-direct|=${abs_direct_delta} |cpu-fmmgpu|=${abs_fmm_delta}"
echo "  speedup:    cpu/direct-gpu=${cpu_vs_direct}x cpu/fmm-gpu=${cpu_vs_fmm_gpu}x direct-gpu/fmm-gpu=${direct_vs_fmm_gpu}x"
echo
echo "CPU log:        $cpu_log"
echo "Direct GPU log: $direct_log"
echo "FMM GPU log:    $fmm_gpu_log"
