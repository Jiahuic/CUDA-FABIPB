#!/usr/bin/env sh
set -eu

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 <mesh_calibration.csv> <msms|nanoshaper> <target_kept_panels> [panel]" >&2
  echo "Example: $0 results/mesh_calibration/.../mesh_calibration.csv msms 5000 test_proteins/1ajj" >&2
  exit 2
fi

csv="$1"
backend="$2"
target="$3"
panel_filter="${4:-}"

if [ ! -f "$csv" ]; then
  echo "Error: calibration CSV not found: $csv" >&2
  exit 2
fi

case "$backend" in
  msms|nanoshaper) ;;
  *)
    echo "Error: backend must be msms or nanoshaper" >&2
    exit 2
    ;;
esac

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT HUP INT TERM

awk -F, -v backend="$backend" -v panel_filter="$panel_filter" '
  NR == 1 { next }
  $2 != backend { next }
  panel_filter != "" && $1 != panel_filter { next }
  {
    printf "%s,%s,%s,%s,%s,%s\n", $1, $4, $7, $8, $12, $13
  }
' "$csv" | sort -t, -k2,2n >"$tmp"

if [ ! -s "$tmp" ]; then
  echo "Error: no matching calibration rows found" >&2
  exit 1
fi

awk -F, -v backend="$backend" -v target="$target" '
  {
    panel = $1
    res = $2 + 0.0
    param_name = $3
    param_value = $4 + 0.0
    kept = $5 + 0
    area = $6 + 0.0

    diff = kept - target
    if (diff < 0) diff = -diff
    if (!best_set || diff < best_diff) {
      best_set = 1
      best_diff = diff
      best_panel = panel
      best_res = res
      best_param_name = param_name
      best_param_value = param_value
      best_kept = kept
      best_area = area
    }

    if (count > 0 && kept > prev_kept) {
      violations++
    }
    prev_kept = kept
    count++

    if (kept <= target && (!low_set || kept > low_kept)) {
      low_set = 1
      low_panel = panel
      low_res = res
      low_param_name = param_name
      low_param_value = param_value
      low_kept = kept
      low_area = area
    }
    if (kept >= target && (!high_set || kept < high_kept)) {
      high_set = 1
      high_panel = panel
      high_res = res
      high_param_name = param_name
      high_param_value = param_value
      high_kept = kept
      high_area = area
    }
  }
  END {
    print "backend,panel,target_kept_panels,estimate_type,mesh_resolution,backend_param_name,backend_param_value,matched_or_estimated_kept_panels,abs_kept_diff,lower_kept_panels,upper_kept_panels"

    if (violations == 0 && low_set && high_set && low_kept != high_kept && low_param_name == high_param_name) {
      t = (target - low_kept) / (high_kept - low_kept)
      est_res = low_res + t * (high_res - low_res)
      est_param = low_param_value + t * (high_param_value - low_param_value)
      printf "%s,%s,%s,interpolated,%.6f,%s,%.6f,%s,%s,%d,%d\n",
        backend, low_panel, target, est_res, low_param_name, est_param, target, 0, low_kept, high_kept
    } else {
      estimate_type = (violations == 0) ? "nearest" : "nearest_nonmonotone"
      printf "%s,%s,%s,%s,%.6f,%s,%.6f,%d,%d,%s,%s\n",
        backend, best_panel, target, estimate_type, best_res, best_param_name, best_param_value,
        best_kept, best_diff,
        low_set ? low_kept : "",
        high_set ? high_kept : ""
    }
  }
' "$tmp"
