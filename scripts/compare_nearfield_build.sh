#!/bin/sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  exit 1
fi

panel="$1"
shift || true

if [ ! -x "$BUILD_DIR/fabipb" ]; then
  echo "error: $BUILD_DIR/fabipb not found; build first with: cmake --build $BUILD_DIR" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/nearfield_build_logs/$timestamp}"
mkdir -p "$OUT_DIR"

prep_log="$OUT_DIR/prep.log"
base_log="$OUT_DIR/baseline.log"
disjoint_log="$OUT_DIR/disjoint_q1.log"

mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 "$panel" "$@" >"$prep_log" 2>&1
fi

./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -m=0 "$panel" "$@" >"$base_log" 2>&1
FABIPB_GPU_NEARFIELD_BUILD_DISJOINT=1 \
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -m=0 "$panel" "$@" >"$disjoint_log" 2>&1

extract_metric() {
  log="$1"
  key="$2"
  case "$key" in
    ttl)
      awk -F'[ ,=]+' '/ttl time:/{val=$3} END{print val}' "$log"
      ;;
    its)
      awk -F'[ ,=]+' '/ttl time:/{val=$5} END{print val}' "$log"
      ;;
    energy)
      awk '/solvation energy:/{val=$3} END{print val}' "$log"
      ;;
    gmres)
      awk -F'[ =]+' '/Top-level stage times \(s\):/{for(i=1;i<=NF;i++){if($i=="gmres"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    near_build)
      awk -F'[ =]+' '/GPU nearfield breakdown \(s\):/{for(i=1;i<=NF;i++){if($i=="build"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    near_coeff)
      awk -F'[ =]+' '/GPU nearfield build breakdown \(s\):/{for(i=1;i<=NF;i++){if($i=="coeff"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    near_upload)
      awk -F'[ =]+' '/GPU nearfield build breakdown \(s\):/{for(i=1;i<=NF;i++){if($i=="upload"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    near_ms)
      awk '/FMM stage avg\/call \(ms\):/{for(i=1;i<=NF;i++){if($i ~ /^Near=/){split($i,a,"="); val=a[2]}}} END{print val}' "$log"
      ;;
    disjoint_mode)
      awk '/GPU nearfield stage2 mode:/{val="yes"} END{if(val=="") val="no"; print val}' "$log"
      ;;
  esac
}

baseline_ttl="$(extract_metric "$base_log" ttl)"
baseline_its="$(extract_metric "$base_log" its)"
baseline_energy="$(extract_metric "$base_log" energy)"
baseline_gmres="$(extract_metric "$base_log" gmres)"
baseline_build="$(extract_metric "$base_log" near_build)"
baseline_coeff="$(extract_metric "$base_log" near_coeff)"
baseline_upload="$(extract_metric "$base_log" near_upload)"
baseline_near_ms="$(extract_metric "$base_log" near_ms)"

disjoint_ttl="$(extract_metric "$disjoint_log" ttl)"
disjoint_its="$(extract_metric "$disjoint_log" its)"
disjoint_energy="$(extract_metric "$disjoint_log" energy)"
disjoint_gmres="$(extract_metric "$disjoint_log" gmres)"
disjoint_build="$(extract_metric "$disjoint_log" near_build)"
disjoint_coeff="$(extract_metric "$disjoint_log" near_coeff)"
disjoint_upload="$(extract_metric "$disjoint_log" near_upload)"
disjoint_near_ms="$(extract_metric "$disjoint_log" near_ms)"
disjoint_mode="$(extract_metric "$disjoint_log" disjoint_mode)"

ttl_speedup="$(awk -v a="$baseline_ttl" -v b="$disjoint_ttl" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
build_speedup="$(awk -v a="$baseline_build" -v b="$disjoint_build" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
coeff_speedup="$(awk -v a="$baseline_coeff" -v b="$disjoint_coeff" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
near_speedup="$(awk -v a="$baseline_near_ms" -v b="$disjoint_near_ms" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
energy_delta="$(awk -v a="$baseline_energy" -v b="$disjoint_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"

echo "Nearfield build comparison"
echo "  panel: $panel"
if [ -f "$prep_log" ]; then
  echo "  prep:  $prep_log"
fi
echo "  baseline: ttl=${baseline_ttl}s its=${baseline_its} energy=${baseline_energy} gmres=${baseline_gmres}s near-build=${baseline_build}s coeff=${baseline_coeff}s upload=${baseline_upload}s near-ms=${baseline_near_ms}"
echo "  disjoint: ttl=${disjoint_ttl}s its=${disjoint_its} energy=${disjoint_energy} gmres=${disjoint_gmres}s near-build=${disjoint_build}s coeff=${disjoint_coeff}s upload=${disjoint_upload}s near-ms=${disjoint_near_ms} enabled=${disjoint_mode}"
echo "  delta:    |energy|=${energy_delta}"
echo "  speedup:  ttl=${ttl_speedup}x near-build=${build_speedup}x coeff=${coeff_speedup}x near-ms=${near_speedup}x"
echo
echo "Baseline log: $base_log"
echo "Disjoint log: $disjoint_log"
