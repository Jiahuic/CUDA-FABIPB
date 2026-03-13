#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
DEPTHS="${DEPTHS:-5 6 7 8}"
SETUP_THREADS="${FABIPB_SETUP_THREADS:-1}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/1a63 -p=9" >&2
  exit 2
fi

panel="$1"
shift
solver_args="$*"
case "$panel" in
  *.pqr) panel="${panel%.pqr}" ;;
esac

if [ ! -x "$BUILD_DIR/fabipb" ]; then
  echo "Error: $BUILD_DIR/fabipb not found. Build first:" >&2
  echo "  cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$BUILD_DIR/m2l_near_logs/$timestamp}"
mkdir -p "$OUT_DIR"

prep_log="$OUT_DIR/prep.log"
mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  FABIPB_SETUP_THREADS="$SETUP_THREADS" \
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g=0 "$panel" >"$prep_log" 2>&1
fi

run_case() {
  mode="$1"
  depth="$2"
  log="$OUT_DIR/${mode}_t${depth}.log"
  if [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$SETUP_THREADS" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g="$mode" -m=0 -t="$depth" "$panel" $solver_args >"$log" 2>&1
  else
    FABIPB_SETUP_THREADS="$SETUP_THREADS" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g="$mode" -m=0 -t="$depth" "$panel" >"$log" 2>&1
  fi
  echo "$log"
}

extract_metric() {
  log="$1"
  key="$2"
  awk -v key="$key" '
    /ttl time:/ {
      for (i = 1; i <= NF; i++) gsub(",", "", $i)
      if (key == "ttl") val = $3
      if (key == "its") { split($4, a, "="); val = a[2] }
    }
    /solvation energy:/ {
      if (key == "energy") val = $3
    }
    /Top-level stage times/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^setupFMM=/ && key == "setupFMM") { split($i, a, "="); val = a[2] }
        if ($i ~ /^gmres=/ && key == "gmres") { split($i, a, "="); val = a[2] }
      }
    }
    /FMM matvec stats:/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^applyFMM=/ && key == "applyFMM") { split($i, a, "="); val = a[2] }
      }
    }
    /FMM stage totals/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^M2L=/ && key == "M2L") { split($i, a, "="); val = a[2] }
        if ($i ~ /^Near=/ && key == "Near") { split($i, a, "="); val = a[2] }
      }
    }
    END {
      if (val == "") val = "NA"
      print val
    }
  ' "$log"
}

ratio() {
  num="$1"
  den="$2"
  awk -v a="$num" -v b="$den" 'BEGIN{if(a=="NA"||b=="NA"||b==0){print "NA"}else{printf "%.6f", a/b}}'
}

echo "M2L/Near comparison"
echo "  panel: $panel"
echo "  depths: $DEPTHS"
echo "  FABIPB_SETUP_THREADS: $SETUP_THREADS"
if [ -f "$prep_log" ]; then
  echo "  prep: $prep_log"
fi
echo

summary="$OUT_DIR/summary.tsv"
cat >"$summary" <<'EOF'
depth	cpu_ttl	gpu_ttl	cpu_gmres	gpu_gmres	cpu_applyFMM	gpu_applyFMM	cpu_M2L	gpu_M2L	cpu_Near	gpu_Near	cpu_its	gpu_its	cpu_energy	gpu_energy	speedup_ttl	speedup_applyFMM	speedup_M2L	speedup_Near
EOF

for depth in $DEPTHS; do
  echo "  running depth $depth CPU"
  cpu_log="$(run_case 0 "$depth")"
  echo "  running depth $depth GPU"
  gpu_log="$(run_case 1 "$depth")"

  cpu_ttl="$(extract_metric "$cpu_log" ttl)"
  gpu_ttl="$(extract_metric "$gpu_log" ttl)"
  cpu_gmres="$(extract_metric "$cpu_log" gmres)"
  gpu_gmres="$(extract_metric "$gpu_log" gmres)"
  cpu_apply="$(extract_metric "$cpu_log" applyFMM)"
  gpu_apply="$(extract_metric "$gpu_log" applyFMM)"
  cpu_m2l="$(extract_metric "$cpu_log" M2L)"
  gpu_m2l="$(extract_metric "$gpu_log" M2L)"
  cpu_near="$(extract_metric "$cpu_log" Near)"
  gpu_near="$(extract_metric "$gpu_log" Near)"
  cpu_its="$(extract_metric "$cpu_log" its)"
  gpu_its="$(extract_metric "$gpu_log" its)"
  cpu_energy="$(extract_metric "$cpu_log" energy)"
  gpu_energy="$(extract_metric "$gpu_log" energy)"

  speedup_ttl="$(ratio "$cpu_ttl" "$gpu_ttl")"
  speedup_apply="$(ratio "$cpu_apply" "$gpu_apply")"
  speedup_m2l="$(ratio "$cpu_m2l" "$gpu_m2l")"
  speedup_near="$(ratio "$cpu_near" "$gpu_near")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$depth" "$cpu_ttl" "$gpu_ttl" "$cpu_gmres" "$gpu_gmres" \
    "$cpu_apply" "$gpu_apply" "$cpu_m2l" "$gpu_m2l" \
    "$cpu_near" "$gpu_near" "$cpu_its" "$gpu_its" \
    "$cpu_energy" "$gpu_energy" "$speedup_ttl" "$speedup_apply" \
    "$speedup_m2l" "$speedup_near" >>"$summary"

  echo "    depth $depth: ttl ${cpu_ttl}s -> ${gpu_ttl}s, applyFMM ${cpu_apply}s -> ${gpu_apply}s, M2L ${cpu_m2l}s -> ${gpu_m2l}s, Near ${cpu_near}s -> ${gpu_near}s"
done

echo
echo "Summary table: $summary"
echo "Logs: $OUT_DIR"
