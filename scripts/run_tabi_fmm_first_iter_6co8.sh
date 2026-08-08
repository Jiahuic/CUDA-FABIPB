#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

pqr="${1:-$repo_root/test_proteins/ZIKV_6CO8_zenodo.pqr}"
tabipb_bin="${TABIPB_BIN:-$repo_root/TABI-PB/build/bin/tabipb}"
nanoshaper_bin="${TABIPB_NANOSHAPER:-$repo_root/TABI-PB/build/bin/NanoShaper}"
fabipb_bin="${FABIPB_BIN:-$repo_root/build/fabipb}"
timestamp="$(date -u +%Y%m%d_%H%M%S)"
out_dir="${OUT_DIR:-$repo_root/results/debug/tabi_fmm_first_iter_6co8/$timestamp}"

sdens="${SDENS:-1}"
fmm_r="${FMM_R:-1}"
pdie="${PDIE:-4}"
sdie="${SDIE:-80}"
bulk="${BULK:-0.15}"
temperature="${TEMPERATURE:-300}"
tree_degree="${TABI_TREE_DEGREE:-3}"
tree_theta="${TABI_TREE_THETA:-0.8}"
tree_leaf="${TABI_TREE_LEAF:-500}"
restart="${GMRES_RESTART:-2}"
tolerance="${GMRES_TOLERANCE:-1e-4}"
initial="${GMRES_INITIAL:-zero}"
dump_stride="${GMRES_DUMP_STRIDE:-1000}"
fmm_depth="${FMM_DEPTH:-8}"
fmm_gpu="${FMM_GPU:-0}"
fmm_q2m="${FMM_Q2M:-0}"
fmm_qorder="${FMM_QORDER:-1}"
fmm_force_tree_rhs="${FMM_FORCE_TREE_RHS:-1}"
fmm_rhs_threads="${FABIPB_RHS_THREADS:-${FMM_RHS_THREADS:-$(nproc)}}"
fmm_rhs_tree_theta="${FABIPB_RHS_TREE_THETA:-${FMM_RHS_TREE_THETA:-0.2}}"
max_fmm_panels="${MAX_FMM_PANELS:-0}"
allow_large_fmm="${ALLOW_LARGE_FMM:-1}"
fabipb_timeout="${FABIPB_TIMEOUT:-}"
live_log="${LIVE_LOG:-1}"

for executable in "$tabipb_bin" "$nanoshaper_bin" "$fabipb_bin"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable is missing: $executable" >&2
    exit 1
  fi
done
if [[ ! -f "$pqr" ]]; then
  echo "PQR is missing: $pqr" >&2
  exit 1
fi
if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -print -quit)" ]] \
  && [[ "${ALLOW_EXISTING_OUT_DIR:-0}" != "1" ]]; then
  echo "Output directory is not empty: $out_dir" >&2
  echo "Choose a new OUT_DIR or set ALLOW_EXISTING_OUT_DIR=1." >&2
  exit 1
fi

mkdir -p "$out_dir"/{tabipb,fmm,compare}
out_dir="$(cd "$out_dir" && pwd)"
tabi_dir="$out_dir/tabipb"
fmm_dir="$out_dir/fmm"
pqr_abs="$(readlink -f "$pqr")"
fmm_rhs_summary="$fmm_dir/rhs_summary.csv"
fmm_rhs_sample="$fmm_dir/rhs_sample.csv"

cat >"$out_dir/run_config.txt" <<EOF
pqr=$pqr_abs
sdens=$sdens
fmm_r=$fmm_r
pdie=$pdie
sdie=$sdie
bulk=$bulk
temperature=$temperature
tabi_tree_degree=$tree_degree
tabi_tree_theta=$tree_theta
tabi_tree_leaf=$tree_leaf
gmres_restart=$restart
gmres_max_iter=1
gmres_tolerance=$tolerance
gmres_initial=$initial
gmres_dump_stride=$dump_stride
fmm_depth=$fmm_depth
fmm_gpu=$fmm_gpu
fmm_q2m=$fmm_q2m
fmm_preconditioner=3
fmm_qorder=$fmm_qorder
fmm_force_tree_rhs=$fmm_force_tree_rhs
fmm_rhs_threads=$fmm_rhs_threads
fmm_rhs_tree_theta=$fmm_rhs_tree_theta
fmm_rhs_summary=$fmm_rhs_summary
fmm_rhs_sample=$fmm_rhs_sample
fmm_rhs_sample_stride=$dump_stride
max_fmm_panels=$max_fmm_panels
allow_large_fmm=$allow_large_fmm
fabipb_timeout=$fabipb_timeout
live_log=$live_log
EOF

ln -sfn "$nanoshaper_bin" "$tabi_dir/NanoShaper"
cat >"$tabi_dir/tabipb.in" <<EOF
mol               $pqr_abs
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
gmres_max_iter    1
gmres_tolerance   $tolerance
precondition      false
outdata           timers
output_prefix     tabipb_first_iter
EOF

echo "[first-iter] TABI-PB"
(
  cd "$tabi_dir"
  tabipb_cmd=(stdbuf -oL -eL "$tabipb_bin" tabipb.in)
  if [[ "$live_log" == "1" ]]; then
    TABIPB_KEEP_MESH=1 \
    TABIPB_STOP_AFTER_GMRES=1 \
    TABIPB_GMRES_INITIAL="$initial" \
    TABIPB_GMRES_STOP_AFTER_ITER=1 \
    TABIPB_GMRES_DUMP_PREFIX="$tabi_dir/gmres" \
    TABIPB_GMRES_DUMP_STRIDE="$dump_stride" \
      "${tabipb_cmd[@]}" 2>&1 | tee tabipb.log
  else
    TABIPB_KEEP_MESH=1 \
    TABIPB_STOP_AFTER_GMRES=1 \
    TABIPB_GMRES_INITIAL="$initial" \
    TABIPB_GMRES_STOP_AFTER_ITER=1 \
    TABIPB_GMRES_DUMP_PREFIX="$tabi_dir/gmres" \
    TABIPB_GMRES_DUMP_STRIDE="$dump_stride" \
      "${tabipb_cmd[@]}" >tabipb.log 2>&1
  fi
)

mv "$tabi_dir/triangulatedSurf.face" "$out_dir/input.face"
mv "$tabi_dir/triangulatedSurf.vert" "$out_dir/input.vert"
ln -sfn "$pqr_abs" "$out_dir/input.pqr"
ln -sfn "$out_dir/input.pqr" "$fmm_dir/input.pqr"
ln -sfn "$out_dir/input.face" "$fmm_dir/input.face"
ln -sfn "$out_dir/input.vert" "$fmm_dir/input.vert"

sha256sum "$out_dir/input.pqr" "$out_dir/input.face" "$out_dir/input.vert" \
  >"$out_dir/input_hashes.txt"

mesh_vertices="$(awk 'NR == 3 { print $1; exit }' "$out_dir/input.vert")"
mesh_faces="$(awk 'NR == 3 { print $1; exit }' "$out_dir/input.face")"
cat >"$out_dir/mesh_counts.txt" <<EOF
vertices=$mesh_vertices
faces=$mesh_faces
EOF
echo "[first-iter] mesh: vertices=$mesh_vertices faces=$mesh_faces"

if [[ "$allow_large_fmm" != "1" ]] && [[ "$max_fmm_panels" != "0" ]] \
  && (( mesh_faces > max_fmm_panels )); then
  cat >"$out_dir/FABIPB_SKIPPED.txt" <<EOF
FABIPB was not run.

The TABI-PB mesh has $mesh_faces faces, which exceeds MAX_FMM_PANELS=$max_fmm_panels.
The current FABIPB Galerkin FMM path expands each face into panel data and then
builds large FMM/M2L/nearfield layouts before the first GMRES matvec. For full
ZIKV_6CO8 sdens=1 this can take far longer than an interactive debug run.

Use a coarser shared TABI/FABIPB mesh, for example:

  SDENS=0.125 FMM_GPU=1 OUT_DIR=<new-dir> $0 $pqr_abs

Or intentionally restore the uncapped large-case default:

  MAX_FMM_PANELS=0 ALLOW_LARGE_FMM=1 FABIPB_TIMEOUT=2h FMM_GPU=1 OUT_DIR=<new-dir> $0 $pqr_abs
EOF
  echo "[first-iter] FABIPB skipped: faces=$mesh_faces exceeds MAX_FMM_PANELS=$max_fmm_panels"
  echo "[first-iter] see: $out_dir/FABIPB_SKIPPED.txt"
  exit 2
fi

echo "[first-iter] FABIPB"
(
  cd "$fmm_dir"
  fabipb_cmd=(
    "$repo_root/scripts/with_benchmark_env.sh" stdbuf -oL -eL "$fabipb_bin"
    -B=1 -g="$fmm_gpu" -m=2 -R="$fmm_r" -t="$fmm_depth"
    -eps1="$pdie" -eps2="$sdie" -P=3 -q="$fmm_qorder" -Q="$fmm_q2m"
    -a="$restart" -i=1 -o="$tolerance" ./input
  )
  if [[ -n "$fabipb_timeout" ]]; then
    fabipb_cmd=(timeout "$fabipb_timeout" "${fabipb_cmd[@]}")
  fi
  if [[ "$live_log" == "1" ]]; then
    FABIPB_REUSE_MESH=1 \
    FABIPB_FORCE_TREE_RHS="$fmm_force_tree_rhs" \
    FABIPB_RHS_THREADS="$fmm_rhs_threads" \
    FABIPB_RHS_TREE_THETA="$fmm_rhs_tree_theta" \
    FABIPB_RHS_SUMMARY_PATH="$fmm_rhs_summary" \
    FABIPB_RHS_SAMPLE_PATH="$fmm_rhs_sample" \
    FABIPB_RHS_SAMPLE_STRIDE="$dump_stride" \
    FABIPB_STOP_AFTER_GMRES=1 \
    FABIPB_GMRES_INITIAL="$initial" \
    FABIPB_GMRES_STOP_AFTER_ITER=1 \
    FABIPB_GMRES_DUMP_PREFIX="$fmm_dir/gmres" \
    FABIPB_GMRES_DUMP_STRIDE="$dump_stride" \
      "${fabipb_cmd[@]}" 2>&1 | tee fmm.log
  else
    FABIPB_REUSE_MESH=1 \
    FABIPB_FORCE_TREE_RHS="$fmm_force_tree_rhs" \
    FABIPB_RHS_THREADS="$fmm_rhs_threads" \
    FABIPB_RHS_TREE_THETA="$fmm_rhs_tree_theta" \
    FABIPB_RHS_SUMMARY_PATH="$fmm_rhs_summary" \
    FABIPB_RHS_SAMPLE_PATH="$fmm_rhs_sample" \
    FABIPB_RHS_SAMPLE_STRIDE="$dump_stride" \
    FABIPB_STOP_AFTER_GMRES=1 \
    FABIPB_GMRES_INITIAL="$initial" \
    FABIPB_GMRES_STOP_AFTER_ITER=1 \
    FABIPB_GMRES_DUMP_PREFIX="$fmm_dir/gmres" \
    FABIPB_GMRES_DUMP_STRIDE="$dump_stride" \
      "${fabipb_cmd[@]}" >fmm.log 2>&1
  fi
)

python3 "$repo_root/scripts/compare_first_iter_vectors.py" \
  --left-prefix "$fmm_dir/gmres" \
  --right-prefix "$tabi_dir/gmres" \
  --left-label FABIPB \
  --right-label TABI-PB \
  --left-rhs-summary "$fmm_rhs_summary" \
  --out-dir "$out_dir/compare"

python3 "$repo_root/scripts/analyze_mesh_normals.py" \
  --vert "$out_dir/input.vert" \
  --face "$out_dir/input.face" \
  --rhs-sample "$fmm_rhs_sample" \
  --out-dir "$out_dir/compare"

cat >"$out_dir/README.md" <<EOF
# TABI-PB/FABIPB First-Iteration Debug Run

Input PQR: \`$pqr_abs\`

This run uses TABI-PB to generate and preserve the NanoShaper mesh, then reuses
the exact \`.face/.vert\` files in FABIPB. Both solvers use diagonal
preconditioning intent and \`$initial\` GMRES initial guess. Both stop after the
first GMRES iteration and skip final energy evaluation. FABIPB uses the
tree-accelerated charge-to-surface RHS by default
(\`FMM_FORCE_TREE_RHS=$fmm_force_tree_rhs\`) so the old direct RHS path is not
used for the large 6CO8 run. The RHS tree uses
\`FABIPB_RHS_THREADS=$fmm_rhs_threads\` and
\`FABIPB_RHS_TREE_THETA=$fmm_rhs_tree_theta\`.

Key files:

- \`run_config.txt\`: command settings
- \`input_hashes.txt\`: PQR and mesh hashes
- \`mesh_counts.txt\`: mesh vertex and face counts
- \`tabipb/tabipb.log\`: TABI-PB log
- \`fmm/fmm.log\`: FABIPB log
- \`tabipb/gmres_*.csv\`: TABI-PB vector dumps
- \`fmm/gmres_*.csv\`: FABIPB vector dumps
- \`tabipb/gmres_metadata.txt\` and \`fmm/gmres_metadata.txt\`: full-vector norms
- \`fmm/rhs_summary.csv\`: FABIPB raw integrated RHS and TABI-style area-averaged RHS sums
- \`fmm/rhs_sample.csv\`: sampled FABIPB RHS rows with centroid, normal, area, raw RHS, and area-normalized RHS
- \`compare/summary.csv\`: vector comparison summary
- \`compare/rhs_sum_comparison.csv\`: TABI-PB \`b\` sums compared with FABIPB RHS sums
- \`compare/normal_summary.csv\`: mesh normal agreement diagnostics for component-1 checks
- \`compare/normal_worst_faces.csv\`: lowest-agreement faces by normal metrics
- \`compare/normal_rhs_sample_join.csv\`: sampled component-1 RHS rows joined with normal metrics
- \`all_output_hashes.txt\`: hashes for every tracked output file
EOF

find "$out_dir" -type f ! -name all_output_hashes.txt -print0 \
  | sort -z | xargs -0 sha256sum >"$out_dir/all_output_hashes.txt"

echo "[first-iter] results: $out_dir"
