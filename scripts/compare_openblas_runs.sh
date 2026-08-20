#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
FABIPB_SETUP_THREADS="${FABIPB_SETUP_THREADS:-1}"
FABIPB_PRECOND_APPLY_THREADS="${FABIPB_PRECOND_APPLY_THREADS:-1}"

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <openblas_a_libdir> <openblas_b_libdir> <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 /opt/openblas-0.3.8/lib /opt/openblas-0.3.20/lib test_proteins/1a63 -g=0 -P=0 -m=1 -R=1.0" >&2
  exit 2
fi

OPENBLAS_A_LIBDIR="$1"
OPENBLAS_B_LIBDIR="$2"
panel="$3"
shift 3 || true

if [ ! -x "$BUILD_DIR/fabipb" ]; then
  echo "error: $BUILD_DIR/fabipb not found; build first with: cmake --build $BUILD_DIR" >&2
  exit 1
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/openblas_compare/$timestamp}"
mkdir -p "$OUT_DIR"

run_case() {
  label="$1"
  libdir="$2"
  log="$3"
  shift 3
  LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  FABIPB_SETUP_THREADS="$FABIPB_SETUP_THREADS" \
  FABIPB_PRECOND_APPLY_THREADS="$FABIPB_PRECOND_APPLY_THREADS" \
  FABIPB_GMRES_LOG_RESID=1 \
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 "$panel" "$@" >"$log" 2>&1
  echo "$label,$libdir,$log"
}

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
    setupPC)
      awk -F'[ =]+' '/Top-level stage times \(s\):/{for(i=1;i<=NF;i++){if($i=="setupPC"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    gmres)
      awk -F'[ =]+' '/Top-level stage times \(s\):/{for(i=1;i<=NF;i++){if($i=="gmres"){val=$(i+1)}}} END{print val}' "$log"
      ;;
    last_resid)
      awk -F'[ =]+' '/GMRES residual:/{val=$6} END{print val}' "$log"
      ;;
  esac
}

log_a="$OUT_DIR/openblas_a.log"
log_b="$OUT_DIR/openblas_b.log"
run_case "openblas_a" "$OPENBLAS_A_LIBDIR" "$log_a" "$@" >/dev/null
run_case "openblas_b" "$OPENBLAS_B_LIBDIR" "$log_b" "$@" >/dev/null

ttl_a="$(extract_metric "$log_a" ttl)"
ttl_b="$(extract_metric "$log_b" ttl)"
its_a="$(extract_metric "$log_a" its)"
its_b="$(extract_metric "$log_b" its)"
energy_a="$(extract_metric "$log_a" energy)"
energy_b="$(extract_metric "$log_b" energy)"
setupPC_a="$(extract_metric "$log_a" setupPC)"
setupPC_b="$(extract_metric "$log_b" setupPC)"
gmres_a="$(extract_metric "$log_a" gmres)"
gmres_b="$(extract_metric "$log_b" gmres)"
resid_a="$(extract_metric "$log_a" last_resid)"
resid_b="$(extract_metric "$log_b" last_resid)"
energy_delta="$(awk -v a="$energy_a" -v b="$energy_b" 'BEGIN{d=a-b; if(d<0)d=-d; printf "%.12e", d}')"

echo "OpenBLAS comparison"
echo "  panel: $panel"
echo "  FABIPB_SETUP_THREADS=$FABIPB_SETUP_THREADS FABIPB_PRECOND_APPLY_THREADS=$FABIPB_PRECOND_APPLY_THREADS"
echo "  A: libdir=$OPENBLAS_A_LIBDIR ttl=${ttl_a}s its=${its_a} energy=${energy_a} setupPC=${setupPC_a}s gmres=${gmres_a}s last_resid=${resid_a}"
echo "  B: libdir=$OPENBLAS_B_LIBDIR ttl=${ttl_b}s its=${its_b} energy=${energy_b} setupPC=${setupPC_b}s gmres=${gmres_b}s last_resid=${resid_b}"
echo "  delta: |energy|=${energy_delta}"
echo
echo "A log: $log_a"
echo "B log: $log_b"
echo "Residual traces can be diffed with:"
echo "  grep '^GMRES residual:' \"$log_a\" > \"$OUT_DIR/a_resid.txt\""
echo "  grep '^GMRES residual:' \"$log_b\" > \"$OUT_DIR/b_resid.txt\""
echo "  diff -u \"$OUT_DIR/a_resid.txt\" \"$OUT_DIR/b_resid.txt\""
