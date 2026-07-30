#!/usr/bin/env sh
set -eu

. "$(dirname "$0")/mesh_control.sh"

BUILD_DIR="${BUILD_DIR:-build}"
DEPTHS="${DEPTHS:-5 6 7 8}"
HYBRID_SETUP_THREADS="${HYBRID_SETUP_THREADS:-8}"
HYBRID_PSOLVE_THREADS="${HYBRID_PSOLVE_THREADS:-8}"
REPEATS="${REPEATS:-3}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [solver options...]" >&2
  echo "Example: $0 test_proteins/H1N1" >&2
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

mesh_control_init

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/hybrid_matrix/$timestamp}"
mkdir -p "$OUT_DIR"

raw_repeats_csv="$OUT_DIR/results_raw.csv"
raw_csv="$OUT_DIR/results.csv"
prep_log="$OUT_DIR/mesh_control.txt"
mesh_control_write_summary "$prep_log" "$panel"

cat >"$raw_repeats_csv" <<'EOF'
case_name,depth,config,repeat,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

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
  depth="$1"
  repeat="$2"
  log="$OUT_DIR/hybrid_best_t${depth}_r$(printf "%02d" "$repeat").log"

  if [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$HYBRID_SETUP_THREADS" FABIPB_PRECOND_APPLY_THREADS="$HYBRID_PSOLVE_THREADS" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -Q=0 "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" "$panel" $solver_args >"$log" 2>&1
  else
    FABIPB_SETUP_THREADS="$HYBRID_SETUP_THREADS" FABIPB_PRECOND_APPLY_THREADS="$HYBRID_PSOLVE_THREADS" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -Q=0 "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" "$panel" >"$log" 2>&1
  fi

  echo "$log"
}

append_raw_row() {
  depth="$1"
  repeat="$2"
  log="$3"
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

  printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "hybrid_best" "$repeat" "1" "0" "$HYBRID_SETUP_THREADS" "$values" >>"$raw_repeats_csv"
}

average_metric() {
  depth="$1"
  key="$2"
  sum="0"
  count=0
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    log="$OUT_DIR/hybrid_best_t${depth}_r$(printf "%02d" "$rep").log"
    value="$(extract_metric "$log" "$key")"
    if [ -n "$value" ]; then
      sum="$(awk -v a="$sum" -v b="$value" 'BEGIN{printf "%.12f", a+b}')"
      count=$((count + 1))
    fi
    rep=$((rep + 1))
  done
  if [ "$count" -eq 0 ]; then
    echo ""
  else
    awk -v s="$sum" -v c="$count" 'BEGIN{printf "%.6f", s/c}'
  fi
}

append_avg_row() {
  depth="$1"
  metrics="ttl its energy loadPanel gkInit setupFMM setupPC setupRHS gmres treecode \
setupFMM_leaf setupFMM_cube_alloc setupFMM_layout setupFMM_apply setupFMM_panel_index \
setupFMM_cubes setupFMM_m2l_pairs setupFMM_m2l_groups gmres_matvec gmres_psolve gmres_basis \
gmres_update gmres_residual gmres_other pc_assemble pc_factor pc_solve pc_scatter pc_other \
applyFMM Q2M M2M M2L L2L L2P Near near_build near_h2d near_kernel near_d2h near_meta near_coeff near_upload near_other"

  values=""
  for key in $metrics; do
    value="$(average_metric "$depth" "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "hybrid_best" "1" "0" "$HYBRID_SETUP_THREADS" "$values" >>"$raw_csv"
}

echo "Hybrid-only benchmark matrix"
echo "  panel: $panel"
echo "  depths: $DEPTHS"
echo "  repeats: $REPEATS"
echo "  mesh backend: $MESH_BACKEND"
echo "  mesh control: $MESH_CONTROL_LABEL = $MESH_CONTROL_VALUE"
echo "  hybrid setup threads: $HYBRID_SETUP_THREADS"
echo "  hybrid psolve threads: $HYBRID_PSOLVE_THREADS"
echo "  mesh control log: $prep_log"
echo

for depth in $DEPTHS; do
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    echo "  running t=$depth repeat $rep/$REPEATS hybrid_best"
    log="$(run_case "$depth" "$rep")"
    append_raw_row "$depth" "$rep" "$log"
    rep=$((rep + 1))
  done
  append_avg_row "$depth"
done

echo
echo "Raw repeat CSV: $raw_repeats_csv"
echo "Raw CSV: $raw_csv"
echo "Logs: $OUT_DIR"
