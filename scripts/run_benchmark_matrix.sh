#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
DEPTHS="${DEPTHS:-5 6 7 8}"
HYBRID_SETUP_THREADS="${HYBRID_SETUP_THREADS:-8}"

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
OUT_DIR="${OUT_DIR:-$BUILD_DIR/benchmark_matrix/$timestamp}"
mkdir -p "$OUT_DIR"

raw_csv="$OUT_DIR/results.csv"
summary_csv="$OUT_DIR/summary.csv"
prep_log="$OUT_DIR/prep.log"

mesh_vert="${panel}.vert"
mesh_face="${panel}.face"
if [ ! -f "$mesh_vert" ] || [ ! -f "$mesh_face" ]; then
  echo "Preparing mesh artifacts for $panel ..."
  FABIPB_SETUP_THREADS=1 \
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 "$panel" >"$prep_log" 2>&1
fi

cat >"$raw_csv" <<'EOF'
case_name,depth,config,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

extract_metric() {
  log="$1"
  key="$2"
  awk -v key="$key" '
    /ttl time:/ {
      for (i = 1; i <= NF; i++) gsub(",", "", $i)
      if (key == "ttl") val = $3
      if (key == "its") { split($4,a,"="); val = a[2] }
    }
    /solvation energy:/ {
      if (key == "energy") val = $3
    }
    /Top-level stage times/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^loadPanel=/ && key == "loadPanel") { split($i,a,"="); val = a[2] }
        if ($i ~ /^gkInit=/ && key == "gkInit") { split($i,a,"="); val = a[2] }
        if ($i ~ /^setupFMM=/ && key == "setupFMM") { split($i,a,"="); val = a[2] }
        if ($i ~ /^setupPC=/ && key == "setupPC") { split($i,a,"="); val = a[2] }
        if ($i ~ /^setupRHS=/ && key == "setupRHS") { split($i,a,"="); val = a[2] }
        if ($i ~ /^gmres=/ && key == "gmres") { split($i,a,"="); val = a[2] }
        if ($i ~ /^treecode=/ && key == "treecode") { split($i,a,"="); val = a[2] }
      }
    }
    /setupFMM breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^leaf-transforms=/ && key == "setupFMM_leaf") { split($i,a,"="); val = a[2] }
        if ($i ~ /^cube-alloc=/ && key == "setupFMM_cube_alloc") { split($i,a,"="); val = a[2] }
        if ($i ~ /^layouts=/ && key == "setupFMM_layout") { split($i,a,"="); val = a[2] }
      }
    }
    /setupFMM layout breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^apply=/ && key == "setupFMM_apply") { split($i,a,"="); val = a[2] }
        if ($i ~ /^panel-index=/ && key == "setupFMM_panel_index") { split($i,a,"="); val = a[2] }
        if ($i ~ /^cubes=/ && key == "setupFMM_cubes") { split($i,a,"="); val = a[2] }
        if ($i ~ /^m2l-pairs=/ && key == "setupFMM_m2l_pairs") { split($i,a,"="); val = a[2] }
        if ($i ~ /^m2l-groups=/ && key == "setupFMM_m2l_groups") { split($i,a,"="); val = a[2] }
      }
    }
    /GMRES breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^matvec=/ && key == "gmres_matvec") { split($i,a,"="); val = a[2] }
        if ($i ~ /^psolve=/ && key == "gmres_psolve") { split($i,a,"="); val = a[2] }
        if ($i ~ /^basis=/ && key == "gmres_basis") { split($i,a,"="); val = a[2] }
        if ($i ~ /^update=/ && key == "gmres_update") { split($i,a,"="); val = a[2] }
        if ($i ~ /^residual=/ && key == "gmres_residual") { split($i,a,"="); val = a[2] }
        if ($i ~ /^other=/ && key == "gmres_other") { split($i,a,"="); val = a[2] }
      }
    }
    /Preconditioner breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^assemble=/ && key == "pc_assemble") { split($i,a,"="); val = a[2] }
        if ($i ~ /^factor=/ && key == "pc_factor") { split($i,a,"="); val = a[2] }
        if ($i ~ /^solve=/ && key == "pc_solve") { split($i,a,"="); val = a[2] }
        if ($i ~ /^scatter=/ && key == "pc_scatter") { split($i,a,"="); val = a[2] }
        if ($i ~ /^other=/ && key == "pc_other") { split($i,a,"="); val = a[2] }
      }
    }
    /FMM matvec stats:/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^applyFMM=/ && key == "applyFMM") { split($i,a,"="); val = a[2] }
      }
    }
    /FMM stage totals/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^Q2M=/ && key == "Q2M") { split($i,a,"="); val = a[2] }
        if ($i ~ /^M2M=/ && key == "M2M") { split($i,a,"="); val = a[2] }
        if ($i ~ /^M2L=/ && key == "M2L") { split($i,a,"="); val = a[2] }
        if ($i ~ /^L2L=/ && key == "L2L") { split($i,a,"="); val = a[2] }
        if ($i ~ /^L2P=/ && key == "L2P") { split($i,a,"="); val = a[2] }
        if ($i ~ /^Near=/ && key == "Near") { split($i,a,"="); val = a[2] }
      }
    }
    /GPU nearfield breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^build=/ && key == "near_build") { split($i,a,"="); val = a[2] }
        if ($i ~ /^h2d=/ && key == "near_h2d") { split($i,a,"="); val = a[2] }
        if ($i ~ /^kernel=/ && key == "near_kernel") { split($i,a,"="); val = a[2] }
        if ($i ~ /^d2h=/ && key == "near_d2h") { split($i,a,"="); val = a[2] }
      }
    }
    /GPU nearfield build breakdown/ {
      for (i = 1; i <= NF; i++) {
        gsub(",", "", $i)
        if ($i ~ /^meta=/ && key == "near_meta") { split($i,a,"="); val = a[2] }
        if ($i ~ /^coeff=/ && key == "near_coeff") { split($i,a,"="); val = a[2] }
        if ($i ~ /^upload=/ && key == "near_upload") { split($i,a,"="); val = a[2] }
        if ($i ~ /^other=/ && key == "near_other") { split($i,a,"="); val = a[2] }
      }
    }
    END {
      if (val == "") val = ""
      print val
    }
  ' "$log"
}

run_case() {
  config="$1"
  depth="$2"
  log="$OUT_DIR/${config}_t${depth}.log"

  case "$config" in
    cpu_serial)
      gpu_mode=0
      q2m_mode=0
      setup_threads=1
      ;;
    gpu_full)
      gpu_mode=1
      q2m_mode=1
      setup_threads=1
      ;;
    hybrid_best)
      gpu_mode=1
      q2m_mode=0
      setup_threads="$HYBRID_SETUP_THREADS"
      ;;
    *)
      echo "Unknown config: $config" >&2
      exit 2
      ;;
  esac

  if [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" -m=0 -t="$depth" "$panel" $solver_args >"$log" 2>&1
  else
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" -m=0 -t="$depth" "$panel" >"$log" 2>&1
  fi

  echo "$log"
}

append_row() {
  config="$1"
  depth="$2"
  log="$3"
  case "$config" in
    cpu_serial) gpu_mode=0; q2m_mode=0; setup_threads=1 ;;
    gpu_full) gpu_mode=1; q2m_mode=1; setup_threads=1 ;;
    hybrid_best) gpu_mode=1; q2m_mode=0; setup_threads="$HYBRID_SETUP_THREADS" ;;
  esac

  metrics="ttl its energy loadPanel gkInit setupFMM setupPC setupRHS gmres treecode \
setupFMM_leaf setupFMM_cube_alloc setupFMM_layout setupFMM_apply setupFMM_panel_index \
setupFMM_cubes setupFMM_m2l_pairs setupFMM_m2l_groups gmres_matvec gmres_psolve gmres_basis \
gmres_update gmres_residual gmres_other pc_assemble pc_factor pc_solve pc_scatter pc_other \
applyFMM Q2M M2M M2L L2L L2P Near near_build near_h2d near_kernel near_d2h near_meta near_coeff near_upload near_other"

  values=""
  for key in $metrics; do
    value="$(extract_metric "$log" "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "$config" "$gpu_mode" "$q2m_mode" "$setup_threads" "$values" >>"$raw_csv"
}

ratio() {
  num="$1"
  den="$2"
  awk -v a="$num" -v b="$den" 'BEGIN{if(a==""||b==""||b==0){print ""}else{printf "%.6f", a/b}}'
}

cat >"$summary_csv" <<'EOF'
case_name,depth,cpu_ttl,gpu_full_ttl,hybrid_best_ttl,cpu_applyFMM,gpu_full_applyFMM,hybrid_best_applyFMM,cpu_M2L,gpu_full_M2L,hybrid_best_M2L,cpu_Near,gpu_full_Near,hybrid_best_Near,cpu_its,gpu_full_its,hybrid_best_its,speedup_gpu_full_ttl,speedup_hybrid_best_ttl,speedup_gpu_full_applyFMM,speedup_hybrid_best_applyFMM,speedup_gpu_full_M2L,speedup_hybrid_best_M2L,speedup_gpu_full_Near,speedup_hybrid_best_Near
EOF

echo "Benchmark matrix"
echo "  panel: $panel"
echo "  depths: $DEPTHS"
echo "  hybrid setup threads: $HYBRID_SETUP_THREADS"
if [ -f "$prep_log" ]; then
  echo "  prep: $prep_log"
fi
echo

for depth in $DEPTHS; do
  echo "  running depth $depth cpu_serial"
  cpu_log="$(run_case cpu_serial "$depth")"
  append_row cpu_serial "$depth" "$cpu_log"

  echo "  running depth $depth gpu_full"
  gpu_full_log="$(run_case gpu_full "$depth")"
  append_row gpu_full "$depth" "$gpu_full_log"

  echo "  running depth $depth hybrid_best"
  hybrid_log="$(run_case hybrid_best "$depth")"
  append_row hybrid_best "$depth" "$hybrid_log"

  cpu_ttl="$(extract_metric "$cpu_log" ttl)"
  gpu_full_ttl="$(extract_metric "$gpu_full_log" ttl)"
  hybrid_ttl="$(extract_metric "$hybrid_log" ttl)"
  cpu_apply="$(extract_metric "$cpu_log" applyFMM)"
  gpu_full_apply="$(extract_metric "$gpu_full_log" applyFMM)"
  hybrid_apply="$(extract_metric "$hybrid_log" applyFMM)"
  cpu_m2l="$(extract_metric "$cpu_log" M2L)"
  gpu_full_m2l="$(extract_metric "$gpu_full_log" M2L)"
  hybrid_m2l="$(extract_metric "$hybrid_log" M2L)"
  cpu_near="$(extract_metric "$cpu_log" Near)"
  gpu_full_near="$(extract_metric "$gpu_full_log" Near)"
  hybrid_near="$(extract_metric "$hybrid_log" Near)"
  cpu_its="$(extract_metric "$cpu_log" its)"
  gpu_full_its="$(extract_metric "$gpu_full_log" its)"
  hybrid_its="$(extract_metric "$hybrid_log" its)"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "$cpu_ttl" "$gpu_full_ttl" "$hybrid_ttl" \
    "$cpu_apply" "$gpu_full_apply" "$hybrid_apply" \
    "$cpu_m2l" "$gpu_full_m2l" "$hybrid_m2l" \
    "$cpu_near" "$gpu_full_near" "$hybrid_near" \
    "$cpu_its" "$gpu_full_its" "$hybrid_its" \
    "$(ratio "$cpu_ttl" "$gpu_full_ttl")" "$(ratio "$cpu_ttl" "$hybrid_ttl")" \
    "$(ratio "$cpu_apply" "$gpu_full_apply")" "$(ratio "$cpu_apply" "$hybrid_apply")" \
    "$(ratio "$cpu_m2l" "$gpu_full_m2l")" "$(ratio "$cpu_m2l" "$hybrid_m2l")" \
    "$(ratio "$cpu_near" "$gpu_full_near")" "$(ratio "$cpu_near" "$hybrid_near")" >>"$summary_csv"
done

echo
echo "Raw CSV: $raw_csv"
echo "Summary CSV: $summary_csv"
echo "Logs: $OUT_DIR"
