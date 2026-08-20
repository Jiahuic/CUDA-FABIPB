#!/usr/bin/env sh
set -eu

BUILD_DIR="${BUILD_DIR:-build}"
RESOLUTIONS="${RESOLUTIONS:-0.50 0.75 1.00 1.25 1.50 2.00}"
OUT_DIR="${OUT_DIR:-results/mesh_calibration}"
TARGET_KEPT_PANELS="${TARGET_KEPT_PANELS:-}"
TARGET_REFINE_FACTORS="${TARGET_REFINE_FACTORS:-0.90 1.00 1.10}"
TARGET_TOLERANCE="${TARGET_TOLERANCE:-100}"
TARGET_MAX_ITERS="${TARGET_MAX_ITERS:-4}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <panel-base-or-pqr-path> [extra fabipb args...]" >&2
  echo "Example: $0 test_proteins/1ajj" >&2
  echo "Env: RESOLUTIONS='0.5 0.75 1.0 1.25 1.5 2.0' BUILD_DIR=build OUT_DIR=results/mesh_calibration TARGET_KEPT_PANELS=5000" >&2
  exit 2
fi

panel="$1"
shift
case "$panel" in
  *.pqr) panel="${panel%.pqr}" ;;
esac

if [ ! -x "$BUILD_DIR/fabipb" ]; then
  echo "Error: $BUILD_DIR/fabipb not found. Build first:" >&2
  echo "  cmake -S . -B $BUILD_DIR && cmake --build $BUILD_DIR" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUT_DIR/$timestamp"
mkdir -p "$RUN_DIR"

csv="$RUN_DIR/mesh_calibration.csv"
match_csv="$RUN_DIR/mesh_panel_matches.csv"
recommend_csv="$RUN_DIR/mesh_target_recommendations.csv"
refined_csv="$RUN_DIR/mesh_calibration_refined.csv"
refined_recommend_csv="$RUN_DIR/mesh_target_recommendations_refined.csv"
iterative_csv="$RUN_DIR/mesh_calibration_iterative.csv"
iterative_recommend_csv="$RUN_DIR/mesh_target_recommendations_iterative.csv"
monotonicity_csv="$RUN_DIR/mesh_monotonicity_report.csv"
warning_csv="$RUN_DIR/mesh_target_warnings.csv"
health_txt="$RUN_DIR/mesh_calibration_health.txt"
printf "%s\n" \
  "panel,backend,mesh_mode,mesh_resolution,mesh_control_name,mesh_control_value,backend_param_name,backend_param_value,atoms,vertices,faces,kept_panels,area,log_path" >"$csv"
printf "%s\n" \
  "stage,backend,status,violations,points,first_resolution,first_kept_panels,last_resolution,last_kept_panels" >"$monotonicity_csv"
printf "%s\n" \
  "stage,backend,panel,target_kept_panels,estimate_type,mesh_resolution,backend_param_name,backend_param_value,matched_or_estimated_kept_panels,abs_kept_diff,lower_kept_panels,upper_kept_panels" >"$warning_csv"

extract_value() {
  log="$1"
  key="$2"
  awk -v key="$key" '
    /Mesh input:/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^atoms=/ && key == "atoms") { split($i,a,"="); val = a[2] }
        if ($i ~ /^mode=/ && key == "mode") { split($i,a,"="); val = a[2] }
        if ($i ~ /^param=/ && key == "param") { split($i,a,"="); val = a[2] }
      }
    }
    /Mesh control:/ {
      for (i = 3; i <= NF; i++) {
        if ($i ~ /^resolved-/) {
          split($i,a,"=")
          name = a[1]
          sub(/^resolved-/, "", name)
          if (key == "backend_param_name") val = name
          if (key == "backend_param_value") val = a[2]
        } else if (index($i, "=") > 0) {
          split($i,a,"=")
          if (key == "mesh_control_name") val = a[1]
          if (key == "mesh_control_value") val = a[2]
        }
      }
    }
    /Mesh raw counts:/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^vertices=/ && key == "vertices") { split($i,a,"="); val = a[2] }
        if ($i ~ /^faces=/ && key == "faces") { split($i,a,"="); val = a[2] }
      }
    }
    /Mesh filtered panels:/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^kept=/ && key == "kept") { split($i,a,"="); val = a[2] }
        if ($i ~ /^area=/ && key == "area") { split($i,a,"="); val = a[2] }
      }
    }
    END { print val }
  ' "$log"
}

backend_name() {
  case "$1" in
    1) echo "msms" ;;
    2) echo "nanoshaper" ;;
    *) echo "unknown" ;;
  esac
}

backend_mode() {
  case "$1" in
    msms) echo "1" ;;
    nanoshaper) echo "2" ;;
    *) echo "0" ;;
  esac
}

run_mesh_case() {
  mode="$1"
  res="$2"
  log="$3"

  if [ "$#" -gt 3 ]; then
    shift 3
    # shellcheck disable=SC2086
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -M=1 -g=0 -m="$mode" -R="$res" "$panel" "$@" >"$log" 2>&1
  else
    ./scripts/with_benchmark_env.sh "$BUILD_DIR/fabipb" -M=1 -g=0 -m="$mode" -R="$res" "$panel" >"$log" 2>&1
  fi
}

append_csv_row_from_log() {
  out_csv="$1"
  backend="$2"
  mode="$3"
  res="$4"
  log="$5"

  atoms="$(extract_value "$log" atoms)"
  control_name="$(extract_value "$log" mesh_control_name)"
  control_value="$(extract_value "$log" mesh_control_value)"
  param_name="$(extract_value "$log" backend_param_name)"
  param_value="$(extract_value "$log" backend_param_value)"
  vertices="$(extract_value "$log" vertices)"
  faces="$(extract_value "$log" faces)"
  kept="$(extract_value "$log" kept)"
  area="$(extract_value "$log" area)"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$panel" "$backend" "$mode" "$res" "$control_name" "$control_value" \
    "$param_name" "$param_value" "$atoms" "$vertices" "$faces" "$kept" "$area" "$log" >>"$out_csv"

  echo "  R=$res -> $param_name=$param_value vertices=$vertices faces=$faces kept=$kept area=$area"
}

csv_has_resolution() {
  check_csv="$1"
  backend="$2"
  res="$3"
  awk -F, -v backend="$backend" -v r="$res" '
    NR == 1 { next }
    $2 == backend && sprintf("%.6f", $4 + 0.0) == sprintf("%.6f", r + 0.0) { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$check_csv"
}

csv_best_actual() {
  check_csv="$1"
  backend="$2"
  target="$3"
  awk -F, -v backend="$backend" -v target="$target" '
    NR == 1 { next }
    $2 != backend { next }
    {
      kept = $12 + 0
      diff = kept - target
      if (diff < 0) diff = -diff
      if (!best_set || diff < best_diff) {
        best_set = 1
        best_res = $4 + 0.0
        best_param_name = $7
        best_param_value = $8 + 0.0
        best_kept = kept
        best_diff = diff
      }
    }
    END {
      if (best_set) {
        printf "%.6f,%s,%.6f,%d,%d\n", best_res, best_param_name, best_param_value, best_kept, best_diff
      }
    }
  ' "$check_csv"
}

csv_row_for_resolution() {
  check_csv="$1"
  backend="$2"
  res="$3"
  awk -F, -v backend="$backend" -v r="$res" '
    NR == 1 { next }
    $2 == backend && sprintf("%.6f", $4 + 0.0) == sprintf("%.6f", r + 0.0) {
      kept = $12 + 0
      diff = kept
      printf "%.6f,%s,%.6f,%d,%f\n", $4 + 0.0, $7, $8 + 0.0, kept, $13 + 0.0
      exit
    }
  ' "$check_csv"
}

append_monotonicity_report() {
  check_csv="$1"
  stage="$2"
  awk -F, -v stage="$stage" '
    NR == 1 { next }
    {
      backend = $2
      n[backend]++
      res[backend, n[backend]] = $4 + 0.0
      kept[backend, n[backend]] = $12 + 0
    }
    END {
      split("msms nanoshaper", backends, " ")
      for (b in backends) {
        backend = backends[b]
        if (n[backend] == 0) continue
        for (i = 1; i <= n[backend]; i++) {
          order[i] = i
        }
        for (i = 1; i <= n[backend]; i++) {
          for (j = i + 1; j <= n[backend]; j++) {
            if (res[backend, order[i]] > res[backend, order[j]]) {
              tmp = order[i]
              order[i] = order[j]
              order[j] = tmp
            }
          }
        }
        violations = 0
        for (i = 2; i <= n[backend]; i++) {
          prev = kept[backend, order[i - 1]]
          curr = kept[backend, order[i]]
          if (curr > prev) {
            violations++
          }
        }
        status = (violations == 0) ? "ok" : "warning_nonmonotone"
        first_idx = order[1]
        last_idx = order[n[backend]]
        printf "%s,%s,%s,%d,%d,%.6f,%d,%.6f,%d\n",
          stage, backend, status, violations, n[backend],
          res[backend, first_idx], kept[backend, first_idx],
          res[backend, last_idx], kept[backend, last_idx]
      }
    }
  ' "$check_csv" >>"$monotonicity_csv"
}

append_warning_rows() {
  rec_csv="$1"
  stage="$2"
  awk -F, -v stage="$stage" '
    NR == 1 { next }
    $4 ~ /nonmonotone/ {
      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
        stage, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
    }
  ' "$rec_csv" >>"$warning_csv"
}

echo "Mesh calibration sweep"
echo "  panel: $panel"
echo "  resolutions: $RESOLUTIONS"
echo "  out dir: $RUN_DIR"
echo

for mode in 1 2; do
  backend="$(backend_name "$mode")"
  echo "[$backend]"
  for res in $RESOLUTIONS; do
    log="$RUN_DIR/${backend}_R${res}.log"
    if [ "$#" -gt 0 ]; then
      run_mesh_case "$mode" "$res" "$log" "$@"
    else
      run_mesh_case "$mode" "$res" "$log"
    fi
    append_csv_row_from_log "$csv" "$backend" "$mode" "$res" "$log"
  done
  echo
done

awk -F, '
  NR == 1 { next }
  $2 == "msms" {
    msms_count++
    msms_panel[msms_count] = $1
    msms_res[msms_count] = $4
    msms_param_name[msms_count] = $7
    msms_param_value[msms_count] = $8
    msms_kept[msms_count] = $12 + 0
    next
  }
  $2 == "nanoshaper" {
    nano_count++
    nano_panel[nano_count] = $1
    nano_res[nano_count] = $4
    nano_param_name[nano_count] = $7
    nano_param_value[nano_count] = $8
    nano_kept[nano_count] = $12 + 0
  }
  END {
    print "panel,msms_resolution,msms_param_name,msms_param_value,msms_kept_panels,nanoshaper_resolution,nanoshaper_param_name,nanoshaper_param_value,nanoshaper_kept_panels,abs_kept_diff"
    for (i = 1; i <= msms_count; i++) {
      best = 0
      best_diff = -1
      for (j = 1; j <= nano_count; j++) {
        diff = msms_kept[i] - nano_kept[j]
        if (diff < 0) diff = -diff
        if (best == 0 || diff < best_diff) {
          best = j
          best_diff = diff
        }
      }
      if (best > 0) {
        printf "%s,%s,%s,%s,%d,%s,%s,%s,%d,%d\n",
          msms_panel[i], msms_res[i], msms_param_name[i], msms_param_value[i], msms_kept[i],
          nano_res[best], nano_param_name[best], nano_param_value[best], nano_kept[best], best_diff
      }
    }
  }
' "$csv" >"$match_csv"

echo "Nearest panel-count matches:"
sed -n '1,12p' "$match_csv"
echo
append_monotonicity_report "$csv" "initial"

if [ -n "$TARGET_KEPT_PANELS" ]; then
  {
    ./scripts/fit_mesh_target.sh "$csv" msms "$TARGET_KEPT_PANELS" "$panel"
    ./scripts/fit_mesh_target.sh "$csv" nanoshaper "$TARGET_KEPT_PANELS" "$panel"
  } | awk '
    NR == 1 { header = $0; print; next }
    $0 == header { next }
    { print }
  ' >"$recommend_csv"

  echo "Target-panel recommendations for kept=$TARGET_KEPT_PANELS:"
  sed -n '1,12p' "$recommend_csv"
  echo
  append_warning_rows "$recommend_csv" "initial"

  cp "$csv" "$refined_csv"
  tail -n +2 "$recommend_csv" | while IFS=, read -r backend_name_row panel_name target_kept estimate_type est_res param_name param_value matched_kept abs_diff low_kept high_kept; do
    mode="$(backend_mode "$backend_name_row")"
    if [ "$mode" = "0" ]; then
      continue
    fi
    echo "Refining $backend_name_row around estimated R=$est_res ..."
    for factor in $TARGET_REFINE_FACTORS; do
      refined_res="$(awk -v r="$est_res" -v f="$factor" 'BEGIN{printf "%.6f", r * f}')"
      if awk -v r="$refined_res" 'BEGIN{exit !(r > 0.0)}'; then
        if awk -F, -v backend="$backend_name_row" -v r="$refined_res" '
          NR == 1 { next }
          $2 == backend && sprintf("%.6f", $4 + 0.0) == sprintf("%.6f", r + 0.0) { found = 1 }
          END { exit found ? 0 : 1 }
        ' "$refined_csv"; then
          continue
        fi
        log="$RUN_DIR/${backend_name_row}_refine_R${refined_res}.log"
        if [ "$#" -gt 0 ]; then
          run_mesh_case "$mode" "$refined_res" "$log" "$@"
        else
          run_mesh_case "$mode" "$refined_res" "$log"
        fi
        append_csv_row_from_log "$refined_csv" "$backend_name_row" "$mode" "$refined_res" "$log"
      fi
    done
    echo
  done

  {
    ./scripts/fit_mesh_target.sh "$refined_csv" msms "$TARGET_KEPT_PANELS" "$panel"
    ./scripts/fit_mesh_target.sh "$refined_csv" nanoshaper "$TARGET_KEPT_PANELS" "$panel"
  } | awk '
    NR == 1 { header = $0; print; next }
    $0 == header { next }
    { print }
  ' >"$refined_recommend_csv"

  echo "Refined target-panel recommendations for kept=$TARGET_KEPT_PANELS:"
  sed -n '1,12p' "$refined_recommend_csv"
  echo
  append_monotonicity_report "$refined_csv" "refined"
  append_warning_rows "$refined_recommend_csv" "refined"

  cp "$refined_csv" "$iterative_csv"
  printf "%s\n" \
    "backend,panel,target_kept_panels,final_mesh_resolution,backend_param_name,backend_param_value,actual_kept_panels,abs_kept_diff,iterations,status" >"$iterative_recommend_csv"

  for backend_name_row in msms nanoshaper; do
    mode="$(backend_mode "$backend_name_row")"
    iter=1
    status="max_iters"
    while [ "$iter" -le "$TARGET_MAX_ITERS" ]; do
      fit_line="$(./scripts/fit_mesh_target.sh "$iterative_csv" "$backend_name_row" "$TARGET_KEPT_PANELS" "$panel" | tail -n 1)"
      IFS=, read -r fit_backend fit_panel fit_target estimate_type est_res param_name param_value matched_kept abs_diff low_kept high_kept <<EOF
$fit_line
EOF
      if [ "$estimate_type" = "nearest" ]; then
        if [ -n "$abs_diff" ] && awk -v d="$abs_diff" -v tol="$TARGET_TOLERANCE" 'BEGIN{exit !(d <= tol)}'; then
          status="converged_existing"
        else
          status="stalled_no_bracket"
        fi
        break
      fi

      if csv_has_resolution "$iterative_csv" "$backend_name_row" "$est_res"; then
        status="stalled_duplicate"
        break
      fi

      echo "Iterative refine $backend_name_row iteration $iter/$TARGET_MAX_ITERS at R=$est_res ..."
      log="$RUN_DIR/${backend_name_row}_iter_R${est_res}.log"
      if [ "$#" -gt 0 ]; then
        run_mesh_case "$mode" "$est_res" "$log" "$@"
      else
        run_mesh_case "$mode" "$est_res" "$log"
      fi
      append_csv_row_from_log "$iterative_csv" "$backend_name_row" "$mode" "$est_res" "$log"

      best_line="$(csv_best_actual "$iterative_csv" "$backend_name_row" "$TARGET_KEPT_PANELS")"
      IFS=, read -r best_res best_param_name best_param_value best_kept best_diff <<EOF
$best_line
EOF
      if [ -n "$best_diff" ] && awk -v d="$best_diff" -v tol="$TARGET_TOLERANCE" 'BEGIN{exit !(d <= tol)}'; then
        status="converged_sampled"
        break
      fi
      iter=$((iter + 1))
      echo
    done

    best_line="$(csv_best_actual "$iterative_csv" "$backend_name_row" "$TARGET_KEPT_PANELS")"
    IFS=, read -r best_res best_param_name best_param_value best_kept best_diff <<EOF
$best_line
EOF
    printf "%s,%s,%s,%.6f,%s,%.6f,%d,%d,%d,%s\n" \
      "$backend_name_row" "$panel" "$TARGET_KEPT_PANELS" "$best_res" "$best_param_name" "$best_param_value" "$best_kept" "$best_diff" "$iter" "$status" >>"$iterative_recommend_csv"
  done

  echo "Iterative target-panel recommendations for kept=$TARGET_KEPT_PANELS (tol=$TARGET_TOLERANCE):"
  sed -n '1,12p' "$iterative_recommend_csv"
  echo
  append_monotonicity_report "$iterative_csv" "iterative"
  append_warning_rows "$iterative_recommend_csv" "iterative"
fi

echo "Monotonicity report:"
sed -n '1,20p' "$monotonicity_csv"
echo

if [ -n "$TARGET_KEPT_PANELS" ]; then
  echo "Fallback warning summary:"
  sed -n '1,20p' "$warning_csv"
  echo
fi

{
  echo "Mesh calibration health summary"
  echo "panel: $panel"
  echo "resolutions: $RESOLUTIONS"
  echo "target kept panels: ${TARGET_KEPT_PANELS:-none}"
  echo
  echo "Monotonicity:"
  tail -n +2 "$monotonicity_csv"
  if [ -n "$TARGET_KEPT_PANELS" ]; then
    echo
    echo "Final iterative recommendations:"
    tail -n +2 "$iterative_recommend_csv"
    echo
    warning_count="$(awk 'NR > 1 { count++ } END { print count + 0 }' "$warning_csv")"
    echo "fallback warning rows: $warning_count"
    if [ "$warning_count" -gt 0 ]; then
      echo
      echo "Fallback warnings:"
      tail -n +2 "$warning_csv"
    fi
  fi
  echo
  echo "Files:"
  echo "calibration_csv: $csv"
  echo "match_csv: $match_csv"
  echo "monotonicity_csv: $monotonicity_csv"
  if [ -n "$TARGET_KEPT_PANELS" ]; then
    echo "recommendation_csv: $recommend_csv"
    echo "refined_csv: $refined_csv"
    echo "refined_recommendation_csv: $refined_recommend_csv"
    echo "iterative_csv: $iterative_csv"
    echo "iterative_recommendation_csv: $iterative_recommend_csv"
    echo "warning_csv: $warning_csv"
  fi
} >"$health_txt"

echo "Health summary:"
sed -n '1,80p' "$health_txt"
echo

echo "CSV: $csv"
echo "Match CSV: $match_csv"
if [ -n "$TARGET_KEPT_PANELS" ]; then
  echo "Recommendation CSV: $recommend_csv"
  echo "Refined CSV: $refined_csv"
  echo "Refined recommendation CSV: $refined_recommend_csv"
  echo "Iterative CSV: $iterative_csv"
  echo "Iterative recommendation CSV: $iterative_recommend_csv"
fi
echo "Monotonicity CSV: $monotonicity_csv"
if [ -n "$TARGET_KEPT_PANELS" ]; then
  echo "Warning CSV: $warning_csv"
fi
echo "Health summary: $health_txt"
