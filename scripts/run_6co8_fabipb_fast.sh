#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fabipb_bin="${FABIPB_BIN:-$repo_root/build/fabipb}"
pqr="${1:-$repo_root/test_proteins/ZIKV_6CO8_zenodo.pqr}"
out_dir="${OUT_DIR:-$repo_root/results/fmm/6co8_fabipb_fast/$(date -u +%Y%m%d_%H%M%S)}"

sdens="${SDENS:-1}"
fmm_r="$(awk -v density="$sdens" 'BEGIN {
  if (density <= 0) exit 1
  printf "%.12g", 1.0 / density
}')" || {
  echo "SDENS must be a positive number: $sdens" >&2
  exit 1
}
fmm_depth="${FMM_DEPTH:-8}"
if [[ -n "${FMM_GPU+x}" ]]; then
  if [[ "$FMM_GPU" != "-1" && "$FMM_GPU" != "0" && "$FMM_GPU" != "1" ]]; then
    echo "FMM_GPU must be -1, 0, or 1: $FMM_GPU" >&2
    exit 1
  fi
  fmm_gpu="$FMM_GPU"
  fmm_gpu_reason="explicit FMM_GPU=$FMM_GPU"
else
  fmm_gpu=-1
  fmm_gpu_reason="auto: binary selects GPU when CUDA is available"
fi
fmm_qorder="${FMM_QORDER:-1}"
restart="${GMRES_RESTART:-30}"
max_iter="${GMRES_MAX_ITER:-100}"
tolerance="${GMRES_TOLERANCE:-1e-4}"
initial="${GMRES_INITIAL:-zero}"
pdie="${PDIE:-4}"
sdie="${SDIE:-80}"
stop_after_rhs="${STOP_AFTER_RHS:-0}"
stop_after_gmres="${STOP_AFTER_GMRES:-0}"
fabipb_timeout="${FABIPB_TIMEOUT:-}"
live_log="${LIVE_LOG:-1}"
rhs_sample_stride="${FABIPB_RHS_SAMPLE_STRIDE:-${RHS_SAMPLE_STRIDE:-1000}}"

nanoshaper_bin="${FABIPB_NANOSHAPER_BIN:-}"
if [[ -z "$nanoshaper_bin" ]]; then
  if [[ -x "$repo_root/TABI-PB/build/bin/NanoShaper" ]]; then
    nanoshaper_bin="$repo_root/TABI-PB/build/bin/NanoShaper"
  else
    nanoshaper_bin="$repo_root/nanoshaper-master/build/NanoShaper"
  fi
fi

if [[ ! -x "$fabipb_bin" ]]; then
  echo "Missing executable: $fabipb_bin" >&2
  exit 1
fi
if [[ ! -f "$pqr" ]]; then
  echo "PQR is missing: $pqr" >&2
  exit 1
fi
if [[ ! -x "$nanoshaper_bin" ]]; then
  echo "NanoShaper is missing or not executable: $nanoshaper_bin" >&2
  exit 1
fi

pqr_abs="$(readlink -f "$pqr")"
if [[ -n "${FMM_Q2M+x}" ]]; then
  if [[ "$FMM_Q2M" != "0" && "$FMM_Q2M" != "1" ]]; then
    echo "FMM_Q2M must be 0 or 1: $FMM_Q2M" >&2
    exit 1
  fi
  fmm_q2m_override="$FMM_Q2M"
else
  fmm_q2m_override="auto"
fi

if [[ -n "${FMM_PRECONDITIONER+x}" ]]; then
  if ! [[ "$FMM_PRECONDITIONER" =~ ^-?[0-9]+$ ]] ||
     [[ "$FMM_PRECONDITIONER" -lt -1 ]] || [[ "$FMM_PRECONDITIONER" -gt 3 ]]; then
    echo "FMM_PRECONDITIONER must be one of -1, 0, 1, 2, 3: $FMM_PRECONDITIONER" >&2
    exit 1
  fi
  fmm_preconditioner_override="$FMM_PRECONDITIONER"
else
  fmm_preconditioner_override="auto"
fi

if [[ -d "$out_dir" ]] && [[ -n "$(find "$out_dir" -mindepth 1 -print -quit)" ]] \
  && [[ "${ALLOW_EXISTING_OUT_DIR:-0}" != "1" ]]; then
  echo "Output directory is not empty: $out_dir" >&2
  echo "Choose a new OUT_DIR or set ALLOW_EXISTING_OUT_DIR=1." >&2
  exit 1
fi

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
rhs_summary_path="$out_dir/rhs_summary.csv"
rhs_sample_path="$out_dir/rhs_sample.csv"
ln -sfn "$pqr_abs" "$out_dir/input.pqr"

cat >"$out_dir/run_config.txt" <<EOF
pqr=$pqr_abs
fabipb_bin=$fabipb_bin
nanoshaper_bin=$nanoshaper_bin
mesh_backend=nanoshaper
sdens=$sdens
internal_fmm_r=$fmm_r
fmm_depth=$fmm_depth
fmm_gpu=$fmm_gpu
fmm_gpu_reason=$fmm_gpu_reason
fmm_q2m_override=$fmm_q2m_override
fmm_qorder=$fmm_qorder
fmm_preconditioner_override=$fmm_preconditioner_override
gmres_restart=$restart
gmres_max_iter=$max_iter
gmres_tolerance=$tolerance
gmres_initial=$initial
pdie=$pdie
sdie=$sdie
force_tree_rhs=1
rhs_tree_theta_override=${FABIPB_RHS_TREE_THETA:-${FMM_RHS_TREE_THETA:-source-default}}
rhs_summary_path=$rhs_summary_path
rhs_sample_path=$rhs_sample_path
rhs_sample_stride=$rhs_sample_stride
rhs_threads_override=${FABIPB_RHS_THREADS:-${FMM_RHS_THREADS:-source-default}}
energy_threads_override=${FABIPB_ENERGY_THREADS:-${FMM_ENERGY_THREADS:-source-default}}
worker_threads_override=${FABIPB_WORKER_THREADS:-source-default}
m2l_chunk_mib_override=${FABIPB_GPU_M2L_CHUNK_MIB:-source-default}
nearfield_chunk_mib_override=${FABIPB_GPU_NEARFIELD_CHUNK_MIB:-source-default}
nearfield_special_cache_mib_override=${FABIPB_GPU_NEARFIELD_SPECIAL_CACHE_MIB:-source-default}
energy_mode_override=${FABIPB_ENERGY_MODE:-source-default}
reuse_mesh=1
stop_after_rhs=$stop_after_rhs
stop_after_gmres=$stop_after_gmres
fabipb_timeout=$fabipb_timeout
live_log=$live_log
EOF

env_args=(
  FABIPB_NANOSHAPER_BIN="$nanoshaper_bin"
  FABIPB_REUSE_MESH=1
  FABIPB_FORCE_TREE_RHS=1
  FABIPB_RHS_SUMMARY_PATH="$rhs_summary_path"
  FABIPB_RHS_SAMPLE_PATH="$rhs_sample_path"
  FABIPB_RHS_SAMPLE_STRIDE="$rhs_sample_stride"
  FABIPB_GMRES_INITIAL="$initial"
)
if [[ -n "${FMM_RHS_THREADS+x}" && -z "${FABIPB_RHS_THREADS+x}" ]]; then
  env_args+=(FABIPB_RHS_THREADS="$FMM_RHS_THREADS")
fi
if [[ -n "${FMM_RHS_TREE_THETA+x}" && -z "${FABIPB_RHS_TREE_THETA+x}" ]]; then
  env_args+=(FABIPB_RHS_TREE_THETA="$FMM_RHS_TREE_THETA")
fi
if [[ -n "${FMM_ENERGY_THREADS+x}" && -z "${FABIPB_ENERGY_THREADS+x}" ]]; then
  env_args+=(FABIPB_ENERGY_THREADS="$FMM_ENERGY_THREADS")
fi
if [[ -n "${FABIPB_WORKER_THREADS+x}" ]]; then
  env_args+=(
    FABIPB_NEARFIELD_BUILD_THREADS="$FABIPB_WORKER_THREADS"
    FABIPB_SETUP_THREADS="$FABIPB_WORKER_THREADS"
    FABIPB_PRECOND_APPLY_THREADS="$FABIPB_WORKER_THREADS"
    FABIPB_DIRECT_THREADS="$FABIPB_WORKER_THREADS"
  )
fi
if [[ "$stop_after_rhs" == "1" ]]; then
  env_args+=(FABIPB_STOP_AFTER_RHS=1)
fi
if [[ "$stop_after_gmres" == "1" ]]; then
  env_args+=(FABIPB_STOP_AFTER_GMRES=1)
fi

fabipb_cmd=(
  stdbuf -oL -eL "$fabipb_bin"
  -B=1 -g="$fmm_gpu" -m=2 -R="$fmm_r" -t="$fmm_depth"
  -eps1="$pdie" -eps2="$sdie" -q="$fmm_qorder"
  -a="$restart" -i="$max_iter" -o="$tolerance"
)
if [[ "$fmm_q2m_override" != "auto" ]]; then
  fabipb_cmd+=(-Q="$fmm_q2m_override")
fi
if [[ "$fmm_preconditioner_override" != "auto" ]]; then
  fabipb_cmd+=(-P="$fmm_preconditioner_override")
fi
fabipb_cmd+=(./input)
if [[ -n "$fabipb_timeout" ]]; then
  fabipb_cmd=(timeout "$fabipb_timeout" "${fabipb_cmd[@]}")
fi

{
  printf '%q ' env "${env_args[@]}" "${fabipb_cmd[@]}"
  printf '\n'
} >"$out_dir/command.txt"

echo "[6co8-fast] FABIPB tree-RHS run"
echo "[6co8-fast] auto config: gpu=$fmm_gpu ($fmm_gpu_reason)"
echo "[6co8-fast] source policy: q2m=$fmm_q2m_override"
echo "[6co8-fast] source policy: preconditioner=$fmm_preconditioner_override"
(
  cd "$out_dir"
  if [[ "$live_log" == "1" ]]; then
    env "${env_args[@]}" "${fabipb_cmd[@]}" 2>&1 | tee fmm.log
  else
    env "${env_args[@]}" "${fabipb_cmd[@]}" >fmm.log 2>&1
  fi
)

python3 - "$out_dir" <<'PY'
import csv
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
text = (out_dir / "fmm.log").read_text(errors="replace")
row = {"status": "ok", "log": "fmm.log"}

patterns = [
    re.compile(r"Mesh input: panel=(?P<panel>\S+) mesh_atoms=(?P<mesh_atoms>\d+) charge_atoms=(?P<charge_atoms>\d+) mode=(?P<mesh_mode>\S+) param=(?P<mesh_param>\S+)"),
    re.compile(r"Mesh control: (?P<mesh_control_name>[^=]+)=(?P<mesh_control_value>\S+) resolved-(?P<backend_param_name>[^=]+)=(?P<backend_param_value>\S+)"),
    re.compile(r"Mesh raw counts: vertices=(?P<vertices>\d+) faces=(?P<faces>\d+)"),
    re.compile(r"#ele=(?P<input_faces>\d+), Mesh filtered panels: kept=(?P<panels>\d+) area=(?P<area>\S+)"),
    re.compile(r"setupRHS direct pairs: panels=(?P<rhs_panels>\d+) charges=(?P<rhs_charges>\d+) pairs=(?P<rhs_pairs>\d+) limit=(?P<rhs_limit>\d+) mode=(?P<rhs_mode>\S+)(?: theta=(?P<rhs_theta>\S+))?"),
    re.compile(r"setupRHS tree evaluator: threads=(?P<rhs_threads>\d+) panels=(?P<rhs_thread_panels>\d+) theta=(?P<rhs_thread_theta>\S+)"),
    re.compile(r"Charge tree build: (?P<charge_tree_time>\S+) s"),
    re.compile(r"GMRES status: info=(?P<gmres_info>\S+) iterations=(?P<gmres_iterations>\d+) final-residual=(?P<gmres_residual>\S+)"),
    re.compile(r"solvation energy: (?P<energy>\S+)"),
]
for pattern in patterns:
    match = pattern.search(text)
    if match:
        row.update({key: value for key, value in match.groupdict().items() if value is not None})

stage_match = re.search(r"(?:Top-level stage times|Stage times) \(s\): (?P<fields>.*)", text)
if stage_match:
    for token in stage_match.group("fields").split():
        if "=" in token:
            key, value = token.split("=", 1)
            row[key] = value
else:
    row["status"] = "check_log"

columns = [
    "status", "mesh_mode", "mesh_control_value", "backend_param_value",
    "vertices", "faces", "panels", "area", "rhs_pairs", "rhs_mode",
    "rhs_theta", "rhs_threads", "charge_tree_time", "loadPanel", "gkInit", "setupFMM",
    "setupPC", "setupRHS", "gmres", "treecode", "gmres_iterations",
    "gmres_residual", "energy", "log",
]
with (out_dir / "summary.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerow(row)

with (out_dir / "README.md").open("w") as fh:
    fh.write("# 6CO8 FABIPB Fast Run\n\n")
    fh.write("FABIPB-only run on the Zenodo 6CO8 PQR. This runner always sets ")
    fh.write("`FABIPB_FORCE_TREE_RHS=1`, so the expensive direct RHS path is not used.\n\n")
    fh.write("See `summary.csv`, `rhs_summary.csv`, `rhs_sample.csv`, `run_config.txt`, `command.txt`, and `fmm.log`.\n")

print(f"Wrote {out_dir / 'summary.csv'}")
PY

find "$out_dir" -type f ! -name all_output_hashes.txt -print0 \
  | sort -z | xargs -0 sha256sum >"$out_dir/all_output_hashes.txt"

echo "[6co8-fast] results: $out_dir"
