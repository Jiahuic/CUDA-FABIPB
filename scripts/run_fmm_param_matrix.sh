#!/usr/bin/env sh
set -eu

. "$(dirname "$0")/mesh_control.sh"

BUILD_DIR="${BUILD_DIR:-build}"
DEPTHS="${DEPTHS:-5 6 7 8}"
HEIGHTS="${HEIGHTS:-1 2 3}"
SEPRATS="${SEPRATS:-0.6 0.8 1.0}"
ORDER_CASES="${ORDER_CASES:-baseline}"
CONFIGS="${CONFIGS:-cpu_serial gpu_full hybrid_best}"
HYBRID_SETUP_THREADS="${HYBRID_SETUP_THREADS:-8}"
REPEATS="${REPEATS:-10}"
PREP_MESH="${PREP_MESH:-1}"

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

mesh_control_init

order_case_args() {
  case "$1" in
    baseline) echo "" ;;
    fix6) echo "-p=6" ;;
    fix8) echo "-p=8" ;;
    fix10) echo "-p=10" ;;
    var6_exact) echo "-p=-6 -pm=-1" ;;
    var8_exact) echo "-p=-8 -pm=-1" ;;
    var6_pm0) echo "-p=-6 -pm=0" ;;
    var6_pm1) echo "-p=-6 -pm=1" ;;
    var8_pm0) echo "-p=-8 -pm=0" ;;
    var8_pm1) echo "-p=-8 -pm=1" ;;
    *)
      echo "Unknown ORDER_CASE: $1" >&2
      exit 2
      ;;
  esac
}

order_case_values() {
  case "$1" in
    baseline) echo "-1,0" ;;
    fix6) echo "6," ;;
    fix8) echo "8," ;;
    fix10) echo "10," ;;
    var6_exact) echo "-6,-1" ;;
    var8_exact) echo "-8,-1" ;;
    var6_pm0) echo "-6,0" ;;
    var6_pm1) echo "-6,1" ;;
    var8_pm0) echo "-8,0" ;;
    var8_pm1) echo "-8,1" ;;
    *)
      echo "," ;;
  esac
}

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-results/fmm_param_matrix/$timestamp}"
mkdir -p "$OUT_DIR"

raw_repeats_csv="$OUT_DIR/results_raw.csv"
raw_csv="$OUT_DIR/results.csv"
summary_csv="$OUT_DIR/summary.csv"
prep_log="$OUT_DIR/mesh_control.txt"
mesh_control_write_summary "$prep_log" "$panel"

cat >"$raw_repeats_csv" <<'EOF'
case_name,depth,height,seprat,order_case,order_arg,ordermom_arg,config,repeat,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

cat >"$raw_csv" <<'EOF'
case_name,depth,height,seprat,order_case,order_arg,ordermom_arg,config,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

cat >"$summary_csv" <<'EOF'
case_name,depth,height,seprat,order_case,order_arg,ordermom_arg,cpu_ttl,gpu_full_ttl,hybrid_best_ttl,cpu_applyFMM,gpu_full_applyFMM,hybrid_best_applyFMM,cpu_M2L,gpu_full_M2L,hybrid_best_M2L,cpu_Near,gpu_full_Near,hybrid_best_Near,cpu_its,gpu_full_its,hybrid_best_its,speedup_gpu_full_ttl,speedup_hybrid_best_ttl,speedup_gpu_full_applyFMM,speedup_hybrid_best_applyFMM,speedup_gpu_full_M2L,speedup_hybrid_best_M2L,speedup_gpu_full_Near,speedup_hybrid_best_Near
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
  height="$3"
  seprat="$4"
  order_case="$5"
  repeat="$6"
  order_args="$(order_case_args "$order_case")"
  log="$OUT_DIR/${config}_d${depth}_h${height}_s${seprat}_o${order_case}_r$(printf "%02d" "$repeat").log"

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

  if [ -n "$solver_args" ] && [ -n "$order_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" -H="$height" -S="$seprat" "$panel" $order_args $solver_args >"$log" 2>&1
  elif [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" -H="$height" -S="$seprat" "$panel" $solver_args >"$log" 2>&1
  elif [ -n "$order_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" -H="$height" -S="$seprat" "$panel" $order_args >"$log" 2>&1
  else
    FABIPB_SETUP_THREADS="$setup_threads" \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g="$gpu_mode" -Q="$q2m_mode" "$MESH_ARG_MODE" "$MESH_ARG_PARAM" -t="$depth" -H="$height" -S="$seprat" "$panel" >"$log" 2>&1
  fi

  echo "$log"
}

append_raw_row() {
  config="$1"
  depth="$2"
  height="$3"
  seprat="$4"
  order_case="$5"
  repeat="$6"
  log="$7"
  order_vals="$(order_case_values "$order_case")"
  order_arg="${order_vals%,*}"
  ordermom_arg="${order_vals#*,}"

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

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "$height" "$seprat" "$order_case" "$order_arg" "$ordermom_arg" "$config" "$repeat" "$gpu_mode" "$q2m_mode" "$setup_threads" "$values" >>"$raw_repeats_csv"
}

ratio() {
  num="$1"
  den="$2"
  awk -v a="$num" -v b="$den" 'BEGIN{if(a==""||b==""||b==0){print ""}else{printf "%.6f", a/b}}'
}

average_metric() {
  config="$1"
  depth="$2"
  height="$3"
  seprat="$4"
  order_case="$5"
  key="$6"
  sum="0"
  count=0
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    log="$OUT_DIR/${config}_d${depth}_h${height}_s${seprat}_o${order_case}_r$(printf "%02d" "$rep").log"
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
  config="$1"
  depth="$2"
  height="$3"
  seprat="$4"
  order_case="$5"
  order_vals="$(order_case_values "$order_case")"
  order_arg="${order_vals%,*}"
  ordermom_arg="${order_vals#*,}"

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
    value="$(average_metric "$config" "$depth" "$height" "$seprat" "$order_case" "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "$height" "$seprat" "$order_case" "$order_arg" "$ordermom_arg" "$config" "$gpu_mode" "$q2m_mode" "$setup_threads" "$values" >>"$raw_csv"
}

echo "FMM parameter matrix"
echo "  panel: $panel"
echo "  depths: $DEPTHS"
echo "  heights: $HEIGHTS"
echo "  SepRat values: $SEPRATS"
echo "  order cases: $ORDER_CASES"
echo "  configs: $CONFIGS"
echo "  repeats: $REPEATS"
echo "  mesh backend: $MESH_BACKEND"
echo "  mesh control: $MESH_CONTROL_LABEL = $MESH_CONTROL_VALUE"
echo "  hybrid setup threads: $HYBRID_SETUP_THREADS"
echo "  mesh control log: $prep_log"
echo

for depth in $DEPTHS; do
  for height in $HEIGHTS; do
    if [ "$height" -gt "$depth" ]; then
      continue
    fi
    for seprat in $SEPRATS; do
      for order_case in $ORDER_CASES; do
        rep=1
        while [ "$rep" -le "$REPEATS" ]; do
          for config in $CONFIGS; do
            echo "  running d=$depth H=$height S=$seprat order=$order_case repeat $rep/$REPEATS $config"
            log="$(run_case "$config" "$depth" "$height" "$seprat" "$order_case" "$rep")"
            append_raw_row "$config" "$depth" "$height" "$seprat" "$order_case" "$rep" "$log"
          done
          rep=$((rep + 1))
        done

        for config in $CONFIGS; do
          append_avg_row "$config" "$depth" "$height" "$seprat" "$order_case"
        done

        if printf '%s\n' "$CONFIGS" | grep -qx 'cpu_serial' &&
           printf '%s\n' "$CONFIGS" | grep -qx 'gpu_full' &&
           printf '%s\n' "$CONFIGS" | grep -qx 'hybrid_best'; then
          order_vals="$(order_case_values "$order_case")"
          order_arg="${order_vals%,*}"
          ordermom_arg="${order_vals#*,}"
          cpu_ttl="$(average_metric cpu_serial "$depth" "$height" "$seprat" "$order_case" ttl)"
          gpu_ttl="$(average_metric gpu_full "$depth" "$height" "$seprat" "$order_case" ttl)"
          hybrid_ttl="$(average_metric hybrid_best "$depth" "$height" "$seprat" "$order_case" ttl)"
          cpu_apply="$(average_metric cpu_serial "$depth" "$height" "$seprat" "$order_case" applyFMM)"
          gpu_apply="$(average_metric gpu_full "$depth" "$height" "$seprat" "$order_case" applyFMM)"
          hybrid_apply="$(average_metric hybrid_best "$depth" "$height" "$seprat" "$order_case" applyFMM)"
          cpu_m2l="$(average_metric cpu_serial "$depth" "$height" "$seprat" "$order_case" M2L)"
          gpu_m2l="$(average_metric gpu_full "$depth" "$height" "$seprat" "$order_case" M2L)"
          hybrid_m2l="$(average_metric hybrid_best "$depth" "$height" "$seprat" "$order_case" M2L)"
          cpu_near="$(average_metric cpu_serial "$depth" "$height" "$seprat" "$order_case" Near)"
          gpu_near="$(average_metric gpu_full "$depth" "$height" "$seprat" "$order_case" Near)"
          hybrid_near="$(average_metric hybrid_best "$depth" "$height" "$seprat" "$order_case" Near)"
          cpu_its="$(average_metric cpu_serial "$depth" "$height" "$seprat" "$order_case" its)"
          gpu_its="$(average_metric gpu_full "$depth" "$height" "$seprat" "$order_case" its)"
          hybrid_its="$(average_metric hybrid_best "$depth" "$height" "$seprat" "$order_case" its)"

          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$panel" "$depth" "$height" "$seprat" "$order_case" "$order_arg" "$ordermom_arg" \
            "$cpu_ttl" "$gpu_ttl" "$hybrid_ttl" \
            "$cpu_apply" "$gpu_apply" "$hybrid_apply" \
            "$cpu_m2l" "$gpu_m2l" "$hybrid_m2l" \
            "$cpu_near" "$gpu_near" "$hybrid_near" \
            "$cpu_its" "$gpu_its" "$hybrid_its" \
            "$(ratio "$cpu_ttl" "$gpu_ttl")" "$(ratio "$cpu_ttl" "$hybrid_ttl")" \
            "$(ratio "$cpu_apply" "$gpu_apply")" "$(ratio "$cpu_apply" "$hybrid_apply")" \
            "$(ratio "$cpu_m2l" "$gpu_m2l")" "$(ratio "$cpu_m2l" "$hybrid_m2l")" \
            "$(ratio "$cpu_near" "$gpu_near")" "$(ratio "$cpu_near" "$hybrid_near")" >>"$summary_csv"
        fi
      done
    done
  done
done

echo
echo "Raw repeat CSV: $raw_repeats_csv"
echo "Raw CSV: $raw_csv"
echo "Summary CSV: $summary_csv"
echo "Logs: $OUT_DIR"
