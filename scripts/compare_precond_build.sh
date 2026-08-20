#!/bin/sh
set -eu

. "$(dirname "$0")/mesh_control.sh"

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

mesh_control_init

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/precond_build_compare/$timestamp}"
mkdir -p "$OUT_DIR"

prep_log="$OUT_DIR/mesh_control.txt"
base_log="$OUT_DIR/baseline.log"
disjoint_log="$OUT_DIR/disjoint_q1.log"
mesh_control_write_summary "$prep_log" "$panel"

./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 "$MESH_ARG_MODE" "$MESH_ARG_PARAM" "$panel" "$@" >"$base_log" 2>&1
FABIPB_GPU_PRECOND_BUILD_DISJOINT=1 \
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 "$MESH_ARG_MODE" "$MESH_ARG_PARAM" "$panel" "$@" >"$disjoint_log" 2>&1

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
    setupPC)
      awk -F'[ =]+' '/Top-level stage times \(s\):/{for(i=1;i<=NF;i++){if($i=="setupPC"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    psolve)
      awk -F'[ =]+' '/GMRES breakdown \(s\):/{for(i=1;i<=NF;i++){if($i=="psolve"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    pc_disjoint)
      awk -F'[ =]+' '/Preconditioner panelIA0 cases:/{for(i=1;i<=NF;i++){if($i=="disjoint"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    pc_total)
      awk -F'[ =]+' '/Preconditioner panelIA0 cases:/{for(i=1;i<=NF;i++){if($i=="total"){val=$(i+1)}}} END{print val}' "$log"
      ;;
  esac
}

baseline_ttl="$(extract_metric "$base_log" ttl)"
baseline_its="$(extract_metric "$base_log" its)"
baseline_energy="$(extract_metric "$base_log" energy)"
baseline_gmres="$(extract_metric "$base_log" gmres)"
baseline_setupPC="$(extract_metric "$base_log" setupPC)"
baseline_psolve="$(extract_metric "$base_log" psolve)"
baseline_pc_disjoint="$(extract_metric "$base_log" pc_disjoint)"
baseline_pc_total="$(extract_metric "$base_log" pc_total)"

disjoint_ttl="$(extract_metric "$disjoint_log" ttl)"
disjoint_its="$(extract_metric "$disjoint_log" its)"
disjoint_energy="$(extract_metric "$disjoint_log" energy)"
disjoint_gmres="$(extract_metric "$disjoint_log" gmres)"
disjoint_setupPC="$(extract_metric "$disjoint_log" setupPC)"
disjoint_psolve="$(extract_metric "$disjoint_log" psolve)"
disjoint_pc_disjoint="$(extract_metric "$disjoint_log" pc_disjoint)"
disjoint_pc_total="$(extract_metric "$disjoint_log" pc_total)"

ttl_speedup="$(awk -v a="$baseline_ttl" -v b="$disjoint_ttl" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
setupPC_speedup="$(awk -v a="$baseline_setupPC" -v b="$disjoint_setupPC" 'BEGIN{if(b==0||b==""){print "inf"}else{printf "%.6f", a/b}}')"
energy_delta="$(awk -v a="$baseline_energy" -v b="$disjoint_energy" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"

echo "Preconditioner build comparison"
echo "  panel: $panel"
echo "  mesh:  $MESH_BACKEND via $MESH_CONTROL_LABEL=$MESH_CONTROL_VALUE"
echo "  mesh control log: $prep_log"
echo "  baseline: ttl=${baseline_ttl}s its=${baseline_its} energy=${baseline_energy} gmres=${baseline_gmres}s setupPC=${baseline_setupPC}s psolve=${baseline_psolve}s disjoint=${baseline_pc_disjoint}/${baseline_pc_total}"
echo "  disjoint: ttl=${disjoint_ttl}s its=${disjoint_its} energy=${disjoint_energy} gmres=${disjoint_gmres}s setupPC=${disjoint_setupPC}s psolve=${disjoint_psolve}s disjoint=${disjoint_pc_disjoint}/${disjoint_pc_total} enabled=yes"
echo "  delta:    |energy|=${energy_delta}"
echo "  speedup:  ttl=${ttl_speedup}x setupPC=${setupPC_speedup}x"
echo
echo "Baseline log: $base_log"
echo "Disjoint log: $disjoint_log"
