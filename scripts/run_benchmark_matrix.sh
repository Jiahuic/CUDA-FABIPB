#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
DEPTHS="${DEPTHS:-5 6 7 8}"
HYBRID_SETUP_THREADS="${HYBRID_SETUP_THREADS:-8}"
REPEATS="${REPEATS:-10}"
MESH_DENSITY="${MESH_DENSITY:-10}"
DIRECT_APPENDIX="${DIRECT_APPENDIX:-1}"
DIRECT_DEPTH="${DIRECT_DEPTH:-5}"

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
raw_repeats_csv="$OUT_DIR/results_raw.csv"
direct_raw_csv="$OUT_DIR/direct_results_raw.csv"
direct_csv="$OUT_DIR/direct_results.csv"
prep_log="$OUT_DIR/prep.log"

echo "Preparing mesh artifacts for $panel at density $MESH_DENSITY ..."
FABIPB_SETUP_THREADS=1 \
  ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=0 -m=1 -d="$MESH_DENSITY" "$panel" >"$prep_log" 2>&1

cat >"$raw_repeats_csv" <<'EOF'
case_name,depth,config,repeat,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

cat >"$raw_csv" <<'EOF'
case_name,depth,config,gpu_mode,gpu_q2m_mode,setup_threads,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,setupFMM_leaf,setupFMM_cube_alloc,setupFMM_layout,setupFMM_apply,setupFMM_panel_index,setupFMM_cubes,setupFMM_m2l_pairs,setupFMM_m2l_groups,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other,applyFMM,Q2M,M2M,M2L,L2L,L2P,Near,near_build,near_h2d,near_kernel,near_d2h,near_meta,near_coeff,near_upload,near_other
EOF

cat >"$direct_raw_csv" <<'EOF'
case_name,depth,config,repeat,direct_status,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other
EOF

cat >"$direct_csv" <<'EOF'
case_name,depth,config,direct_status,ttl,its,energy,loadPanel,gkInit,setupFMM,setupPC,setupRHS,gmres,treecode,gmres_matvec,gmres_psolve,gmres_basis,gmres_update,gmres_residual,gmres_other,pc_assemble,pc_factor,pc_solve,pc_scatter,pc_other
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
  repeat="$3"
  log="$OUT_DIR/${config}_t${depth}_r$(printf "%02d" "$repeat").log"

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

run_direct_case() {
  repeat="$1"
  log="$OUT_DIR/direct_gpu_t${DIRECT_DEPTH}_r$(printf "%02d" "$repeat").log"

  if [ -n "$solver_args" ]; then
    # shellcheck disable=SC2086
    FABIPB_SETUP_THREADS=1 \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -r=1 -m=0 -t="$DIRECT_DEPTH" "$panel" $solver_args >"$log" 2>&1
  else
    FABIPB_SETUP_THREADS=1 \
      ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -B=1 -g=1 -r=1 -m=0 -t="$DIRECT_DEPTH" "$panel" >"$log" 2>&1
  fi

  echo "$log"
}

direct_status_from_log() {
  log="$1"
  if rg -q "Direct GPU matvec unavailable; using FMM path\\." "$log"; then
    echo "fallback"
  elif rg -q "GPU direct cache: panel-pairs=" "$log"; then
    echo "direct"
  else
    echo "unknown"
  fi
}

append_raw_row() {
  config="$1"
  depth="$2"
  repeat="$3"
  log="$3"
  if [ "$#" -ge 4 ]; then
    log="$4"
  fi
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
    "$panel" "$depth" "$config" "$repeat" "$gpu_mode" "$q2m_mode" "$setup_threads" "$values" >>"$raw_repeats_csv"
}

ratio() {
  num="$1"
  den="$2"
  awk -v a="$num" -v b="$den" 'BEGIN{if(a==""||b==""||b==0){print ""}else{printf "%.6f", a/b}}'
}

average_metric() {
  config="$1"
  depth="$2"
  key="$3"
  sum="0"
  count=0
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    log="$OUT_DIR/${config}_t${depth}_r$(printf "%02d" "$rep").log"
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
    value="$(average_metric "$config" "$depth" "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$depth" "$config" "$gpu_mode" "$q2m_mode" "$setup_threads" "$values" >>"$raw_csv"
}

append_direct_raw_row() {
  repeat="$1"
  log="$2"
  status="$(direct_status_from_log "$log")"
  metrics="ttl its energy loadPanel gkInit setupFMM setupPC setupRHS gmres treecode \
gmres_matvec gmres_psolve gmres_basis gmres_update gmres_residual gmres_other \
pc_assemble pc_factor pc_solve pc_scatter pc_other"

  values=""
  for key in $metrics; do
    value="$(extract_metric "$log" "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$DIRECT_DEPTH" "direct_gpu" "$repeat" "$status" "$values" >>"$direct_raw_csv"
}

average_direct_metric() {
  key="$1"
  sum="0"
  count=0
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    log="$OUT_DIR/direct_gpu_t${DIRECT_DEPTH}_r$(printf "%02d" "$rep").log"
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

append_direct_avg_row() {
  status="unknown"
  if [ "$REPEATS" -ge 1 ]; then
    status="$(direct_status_from_log "$OUT_DIR/direct_gpu_t${DIRECT_DEPTH}_r01.log")"
  fi

  metrics="ttl its energy loadPanel gkInit setupFMM setupPC setupRHS gmres treecode \
gmres_matvec gmres_psolve gmres_basis gmres_update gmres_residual gmres_other \
pc_assemble pc_factor pc_solve pc_scatter pc_other"

  values=""
  for key in $metrics; do
    value="$(average_direct_metric "$key")"
    if [ -n "$values" ]; then
      values="$values,$value"
    else
      values="$value"
    fi
  done

  printf "%s,%s,%s,%s,%s\n" \
    "$panel" "$DIRECT_DEPTH" "direct_gpu" "$status" "$values" >>"$direct_csv"
}

cat >"$summary_csv" <<'EOF'
case_name,depth,cpu_ttl,gpu_full_ttl,hybrid_best_ttl,cpu_applyFMM,gpu_full_applyFMM,hybrid_best_applyFMM,cpu_M2L,gpu_full_M2L,hybrid_best_M2L,cpu_Near,gpu_full_Near,hybrid_best_Near,cpu_its,gpu_full_its,hybrid_best_its,speedup_gpu_full_ttl,speedup_hybrid_best_ttl,speedup_gpu_full_applyFMM,speedup_hybrid_best_applyFMM,speedup_gpu_full_M2L,speedup_hybrid_best_M2L,speedup_gpu_full_Near,speedup_hybrid_best_Near
EOF

echo "Benchmark matrix"
echo "  panel: $panel"
echo "  depths: $DEPTHS"
echo "  repeats: $REPEATS"
echo "  mesh density: $MESH_DENSITY"
echo "  hybrid setup threads: $HYBRID_SETUP_THREADS"
echo "  direct appendix: $DIRECT_APPENDIX"
echo "  direct depth: $DIRECT_DEPTH"
if [ -f "$prep_log" ]; then
  echo "  prep: $prep_log"
fi
echo

for depth in $DEPTHS; do
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    echo "  running depth $depth repeat $rep/$REPEATS cpu_serial"
    cpu_log="$(run_case cpu_serial "$depth" "$rep")"
    append_raw_row cpu_serial "$depth" "$rep" "$cpu_log"

    echo "  running depth $depth repeat $rep/$REPEATS gpu_full"
    gpu_full_log="$(run_case gpu_full "$depth" "$rep")"
    append_raw_row gpu_full "$depth" "$rep" "$gpu_full_log"

    echo "  running depth $depth repeat $rep/$REPEATS hybrid_best"
    hybrid_log="$(run_case hybrid_best "$depth" "$rep")"
    append_raw_row hybrid_best "$depth" "$rep" "$hybrid_log"
    rep=$((rep + 1))
  done

  append_avg_row cpu_serial "$depth"
  append_avg_row gpu_full "$depth"
  append_avg_row hybrid_best "$depth"

  cpu_ttl="$(average_metric cpu_serial "$depth" ttl)"
  gpu_full_ttl="$(average_metric gpu_full "$depth" ttl)"
  hybrid_ttl="$(average_metric hybrid_best "$depth" ttl)"
  cpu_apply="$(average_metric cpu_serial "$depth" applyFMM)"
  gpu_full_apply="$(average_metric gpu_full "$depth" applyFMM)"
  hybrid_apply="$(average_metric hybrid_best "$depth" applyFMM)"
  cpu_m2l="$(average_metric cpu_serial "$depth" M2L)"
  gpu_full_m2l="$(average_metric gpu_full "$depth" M2L)"
  hybrid_m2l="$(average_metric hybrid_best "$depth" M2L)"
  cpu_near="$(average_metric cpu_serial "$depth" Near)"
  gpu_full_near="$(average_metric gpu_full "$depth" Near)"
  hybrid_near="$(average_metric hybrid_best "$depth" Near)"
  cpu_its="$(average_metric cpu_serial "$depth" its)"
  gpu_full_its="$(average_metric gpu_full "$depth" its)"
  hybrid_its="$(average_metric hybrid_best "$depth" its)"

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

if [ "$DIRECT_APPENDIX" -gt 0 ]; then
  rep=1
  while [ "$rep" -le "$REPEATS" ]; do
    echo "  running direct appendix repeat $rep/$REPEATS"
    direct_log="$(run_direct_case "$rep")"
    append_direct_raw_row "$rep" "$direct_log"
    rep=$((rep + 1))
  done
  append_direct_avg_row
fi

echo
echo "Raw repeat CSV: $raw_repeats_csv"
echo "Raw CSV: $raw_csv"
echo "Summary CSV: $summary_csv"
if [ "$DIRECT_APPENDIX" -gt 0 ]; then
  echo "Direct raw CSV: $direct_raw_csv"
  echo "Direct CSV: $direct_csv"
fi
echo "Logs: $OUT_DIR"
