#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <pqr> [pqr ...]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
tabipb_bin="${TABIPB_BIN:-$repo_root/TABI-PB/build/bin/tabipb}"
nanoshaper_bin="${TABIPB_NANOSHAPER:-$repo_root/TABI-PB/build/bin/NanoShaper}"
fabipb_bin="${FABIPB_BIN:-$repo_root/build/fabipb}"
out_dir="${OUT_DIR:-$repo_root/results/comparison/tabi_fmm_size_sweep/$(date -u +%Y%m%d_%H%M%S)}"

sdens="${SDENS:-1}"
pdie="${PDIE:-4}"
sdie="${SDIE:-80}"
bulk="${BULK:-0.15}"
temperature="${TEMPERATURE:-300}"
tree_degree="${TABI_TREE_DEGREE:-3}"
tree_theta="${TABI_TREE_THETA:-0.8}"
tree_leaf="${TABI_TREE_LEAF:-500}"
restart="${GMRES_RESTART:-30}"
max_iter="${GMRES_MAX_ITER:-100}"
tolerance="${GMRES_TOLERANCE:-1e-4}"
fmm_depth="${FMM_DEPTH:-5}"
fmm_preconditioner="${FMM_PRECONDITIONER:--1}"
reuse_existing="${REUSE_EXISTING:-0}"
run_skip_self="${RUN_SKIP_SELF:-1}"

for executable in "$tabipb_bin" "$nanoshaper_bin" "$fabipb_bin"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable is missing: $executable" >&2
    exit 1
  fi
done

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
cat >"$out_dir/run_config.txt" <<EOF
sdens=$sdens
pdie=$pdie
sdie=$sdie
bulk=$bulk
temperature=$temperature
tabi_tree_degree=$tree_degree
tabi_tree_theta=$tree_theta
tabi_tree_leaf=$tree_leaf
gmres_restart=$restart
gmres_max_iter=$max_iter
gmres_tolerance=$tolerance
fmm_depth=$fmm_depth
fmm_preconditioner=$fmm_preconditioner
run_skip_self=$run_skip_self
EOF
summary="$out_dir/summary.csv"
printf '%s\n' \
  'case,atoms,mesh_vertices,mesh_faces,fmm_kept_panels,tabi_area,fmm_area,tabi_iterations,tabi_residual,tabi_energy,fmm_normal_iterations,fmm_normal_residual,fmm_normal_energy,fmm_skip_self_iterations,fmm_skip_self_residual,fmm_skip_self_energy' \
  >"$summary"

csv_field() {
  local file="$1"
  local field="$2"
  awk -F',' -v field="$field" 'NF { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field); value=$field } END { print value }' "$file"
}

fmm_metric() {
  local file="$1"
  local metric="$2"
  awk -v metric="$metric" '
    /#ele=/ && metric == "panels" { split($1,a,"="); value=a[2]; gsub(/,/, "", value) }
    /#ele=/ && metric == "area" { for(i=1;i<=NF;i++) if($i ~ /^area=/){split($i,a,"="); value=a[2]} }
    /GMRES status:/ && metric == "iterations" { for(i=1;i<=NF;i++) if($i ~ /^iterations=/){split($i,a,"="); value=a[2]} }
    /GMRES status:/ && metric == "residual" { for(i=1;i<=NF;i++) if($i ~ /^final-residual=/){split($i,a,"="); value=a[2]} }
    /solvation energy:/ && metric == "energy" { value=$3 }
    END { print value }
  ' "$file"
}

fmm_run_complete() {
  [[ -s "$1" ]] && grep -q 'solvation energy:' "$1"
}

run_fmm() {
  local base="$1"
  local log="$2"
  local skip_self="$3"
  local work_dir
  local input_base
  work_dir="$(dirname "$base")"
  input_base="./$(basename "$base")"
  local -a env_args=(
    FABIPB_REUSE_MESH=1
    FABIPB_FORCE_TREE_RHS=1
  )
  if [[ "$skip_self" == "1" ]]; then
    env_args+=(FABIPB_SKIP_PANEL_CASES=self)
  fi

  (
    cd "$work_dir"
    env "${env_args[@]}" \
      "$repo_root/scripts/with_benchmark_env.sh" "$fabipb_bin" \
        -B=1 -g=0 -m=2 -R=1.0 -t="$fmm_depth" \
        -eps1="$pdie" -eps2="$sdie" -P="$fmm_preconditioner" -q=1 \
        -a="$restart" -i="$max_iter" -o="$tolerance" \
        "$input_base" >"$log" 2>&1
  )
}

for pqr_arg in "$@"; do
  pqr="$(readlink -f "$pqr_arg")"
  if [[ ! -f "$pqr" ]]; then
    echo "PQR is missing: $pqr_arg" >&2
    exit 1
  fi

  case_name="$(basename "$pqr")"
  case_name="${case_name%.pqr}"
  case_dir="$out_dir/$case_name"
  base="$case_dir/input"
  mkdir -p "$case_dir"
  ln -sf "$pqr" "$base.pqr"
  ln -sf "$nanoshaper_bin" "$case_dir/NanoShaper"

  cat >"$case_dir/tabipb.in" <<EOF
mol               $base.pqr
mesh              SES
sdens             $sdens
srad              1.4
pdie              $pdie
sdie              $sdie
bulk              $bulk
temp              $temperature
tree_degree       $tree_degree
tree_max_per_leaf $tree_leaf
tree_theta        $tree_theta
gmres_restart     $restart
gmres_max_iter    $max_iter
gmres_tolerance   $tolerance
outdata           csv
outdata           csv_headers
outdata           timers
output_prefix     tabipb
EOF

  if [[ "$reuse_existing" == "1" &&
        -s "$case_dir/tabipb.csv" &&
        -s "$base.face" && -s "$base.vert" ]] &&
     fmm_run_complete "$case_dir/fmm_normal.log" &&
     { [[ "$run_skip_self" != "1" ]] ||
       fmm_run_complete "$case_dir/fmm_skip_self.log"; }; then
    echo "[$case_name] reusing completed TABI-PB/FMM logs"
  else
    echo "[$case_name] TABI-PB"
    (
      cd "$case_dir"
      TABIPB_KEEP_MESH=1 "$tabipb_bin" tabipb.in >tabipb.log 2>&1
    )
    mv "$case_dir/triangulatedSurf.face" "$base.face"
    mv "$case_dir/triangulatedSurf.vert" "$base.vert"
    sha256sum "$base.face" "$base.vert" >"$case_dir/mesh_sha256_before_fmm.txt"

    echo "[$case_name] FMM normal"
    run_fmm "$base" "$case_dir/fmm_normal.log" 0
    if [[ "$run_skip_self" == "1" ]]; then
      echo "[$case_name] FMM skip-self"
      run_fmm "$base" "$case_dir/fmm_skip_self.log" 1
    fi
    sha256sum "$base.face" "$base.vert" >"$case_dir/mesh_sha256_after_fmm.txt"
    if ! cmp -s "$case_dir/mesh_sha256_before_fmm.txt" \
                  "$case_dir/mesh_sha256_after_fmm.txt"; then
      echo "Mesh changed while running FMM: $case_name" >&2
      exit 1
    fi
  fi

  atoms="$(awk '/^(ATOM|HETATM)/{n++} END{print n+0}' "$pqr")"
  mesh_vertices="$(awk 'NR==3{print $1; exit}' "$base.vert")"
  mesh_faces="$(awk 'NR==3{print $1; exit}' "$base.face")"
  fmm_panels="$(fmm_metric "$case_dir/fmm_normal.log" panels)"
  tabi_area="$(csv_field "$case_dir/tabipb.csv" 10)"
  fmm_area="$(fmm_metric "$case_dir/fmm_normal.log" area)"
  tabi_iterations="$(csv_field "$case_dir/tabipb.csv" 11)"
  tabi_residual="$(csv_field "$case_dir/tabipb.csv" 12)"
  tabi_energy="$(csv_field "$case_dir/tabipb.csv" 13)"
  normal_iterations="$(fmm_metric "$case_dir/fmm_normal.log" iterations)"
  normal_residual="$(fmm_metric "$case_dir/fmm_normal.log" residual)"
  normal_energy="$(fmm_metric "$case_dir/fmm_normal.log" energy)"
  skip_iterations=""
  skip_residual=""
  skip_energy=""
  if [[ -s "$case_dir/fmm_skip_self.log" ]]; then
    skip_iterations="$(fmm_metric "$case_dir/fmm_skip_self.log" iterations)"
    skip_residual="$(fmm_metric "$case_dir/fmm_skip_self.log" residual)"
    skip_energy="$(fmm_metric "$case_dir/fmm_skip_self.log" energy)"
  fi

  printf '%s\n' \
    "$case_name,$atoms,$mesh_vertices,$mesh_faces,$fmm_panels,$tabi_area,$fmm_area,$tabi_iterations,$tabi_residual,$tabi_energy,$normal_iterations,$normal_residual,$normal_energy,$skip_iterations,$skip_residual,$skip_energy" \
    >>"$summary"
  rm -f "$base.xyzr"
done

echo "Comparison summary: $summary"
