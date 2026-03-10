#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
REPEATS="${REPEATS:-10}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/1a63" >&2
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
OUT_DIR="${OUT_DIR:-$BUILD_DIR/nearfield_mode_logs/$timestamp}"
mkdir -p "$OUT_DIR"

prep_log="$OUT_DIR/prep.log"
mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g=0 "$panel" "$@" >"$prep_log" 2>&1
fi

run_mode() {
  mode="$1"
  idx="$2"
  log="$OUT_DIR/mode${mode}_run$(printf "%02d" "$idx").log"
  if [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g=1 -G="$mode" -m=0 "$panel" $solver_args >"$log" 2>&1
  else
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -g=1 -G="$mode" -m=0 "$panel" >"$log" 2>&1
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
    }
    /solvation energy:/ {
      if (key == "energy") val = $3
    }
    /FMM stage totals/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^Near=/ && key == "near") { split($i, a, "="); val = a[2] }
      }
    }
    /GPU nearfield breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^build=/ && key == "build") { split($i, a, "="); val = a[2] }
        if ($i ~ /^h2d=/ && key == "h2d") { split($i, a, "="); val = a[2] }
        if ($i ~ /^kernel=/ && key == "kernel") { split($i, a, "="); val = a[2] }
        if ($i ~ /^d2h=/ && key == "d2h") { split($i, a, "="); val = a[2] }
      }
    }
    END {
      if (val == "") val = "NA"
      print val
    }
  ' "$log"
}

average_metric() {
  mode="$1"
  key="$2"
  sum="0"
  count=0
  idx=1
  while [ "$idx" -le "$REPEATS" ]; do
    log="$OUT_DIR/mode${mode}_run$(printf "%02d" "$idx").log"
    value="$(extract_metric "$log" "$key")"
    if [ "$value" != "NA" ]; then
      sum="$(awk -v a="$sum" -v b="$value" 'BEGIN{printf "%.12f", a+b}')"
      count=$((count + 1))
    fi
    idx=$((idx + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo "NA"
  else
    awk -v s="$sum" -v c="$count" 'BEGIN{printf "%.6f", s/c}'
  fi
}

echo "Nearfield mode comparison"
echo "  panel: $panel"
echo "  repeats: $REPEATS"
if [ -f "$prep_log" ]; then
  echo "  prep: $prep_log"
fi

idx=1
while [ "$idx" -le "$REPEATS" ]; do
  echo "  running mode 0, iteration $idx/$REPEATS"
  run_mode 0 "$idx" >/dev/null
  idx=$((idx + 1))
done

idx=1
while [ "$idx" -le "$REPEATS" ]; do
  echo "  running mode 1, iteration $idx/$REPEATS"
  run_mode 1 "$idx" >/dev/null
  idx=$((idx + 1))
done

mode0_ttl="$(average_metric 0 ttl)"
mode0_energy="$(average_metric 0 energy)"
mode0_near="$(average_metric 0 near)"
mode0_build="$(average_metric 0 build)"
mode0_h2d="$(average_metric 0 h2d)"
mode0_kernel="$(average_metric 0 kernel)"
mode0_d2h="$(average_metric 0 d2h)"

mode1_ttl="$(average_metric 1 ttl)"
mode1_energy="$(average_metric 1 energy)"
mode1_near="$(average_metric 1 near)"
mode1_build="$(average_metric 1 build)"
mode1_h2d="$(average_metric 1 h2d)"
mode1_kernel="$(average_metric 1 kernel)"
mode1_d2h="$(average_metric 1 d2h)"

speedup="$(awk -v a="$mode0_ttl" -v b="$mode1_ttl" 'BEGIN{if(a=="NA"||b=="NA"||b==0){print "NA"}else{printf "%.6f", a/b}}')"
near_ratio="$(awk -v a="$mode0_near" -v b="$mode1_near" 'BEGIN{if(a=="NA"||b=="NA"||b==0){print "NA"}else{printf "%.6f", a/b}}')"

echo
echo "Averages"
echo "  mode 0 (interaction): ttl=${mode0_ttl}s energy=${mode0_energy} near=${mode0_near}s build=${mode0_build}s h2d=${mode0_h2d}s kernel=${mode0_kernel}s d2h=${mode0_d2h}s"
echo "  mode 1 (destination-leaf): ttl=${mode1_ttl}s energy=${mode1_energy} near=${mode1_near}s build=${mode1_build}s h2d=${mode1_h2d}s kernel=${mode1_kernel}s d2h=${mode1_d2h}s"
echo "  comparison: ttl mode0/mode1=${speedup}x near mode0/mode1=${near_ratio}x"
echo
echo "Logs: $OUT_DIR"
