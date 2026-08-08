#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fabipb_bin="${FABIPB_BIN:-$repo_root/build/fabipb}"
pqr="${1:-$repo_root/test_proteins/1a63.pqr}"
out_dir="${OUT_DIR:-$repo_root/results/debug/rhs_density_1a63/$(date -u +%Y%m%d_%H%M%S)}"

default_r="${DEFAULT_R:-1}"
high_density="${HIGH_MSMS_DENSITY:-10}"
fmm_depth="${FMM_DEPTH:-5}"
fmm_gpu="${FMM_GPU:-0}"
pdie="${PDIE:-4}"
sdie="${SDIE:-80}"
q_order="${Q_ORDER:-1}"

if [[ ! -x "$fabipb_bin" ]]; then
  echo "Missing executable: $fabipb_bin" >&2
  exit 1
fi

if ! command -v msms >/dev/null 2>&1; then
  echo "Missing MSMS executable in PATH; this test intentionally uses -m=1." >&2
  exit 1
fi

if [[ ! -f "$pqr" ]]; then
  echo "Missing PQR: $pqr" >&2
  exit 1
fi

mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
pqr="$(readlink -f "$pqr")"

cat >"$out_dir/run_config.txt" <<EOF
pqr=$pqr
fabipb_bin=$fabipb_bin
mesh_backend=msms
default_r=$default_r
high_msms_density=$high_density
fmm_depth=$fmm_depth
fmm_gpu=$fmm_gpu
pdie=$pdie
sdie=$sdie
q_order=$q_order
rhs_modes=default,tree
stop_after_rhs=1
reuse_mesh=1
EOF

run_case() {
  local mesh_case="$1"
  local mesh_arg="$2"
  local rhs_case="$3"
  local case_dir="$out_dir/$mesh_case/$rhs_case"
  local base="$case_dir/input"
  local -a env_args=(FABIPB_STOP_AFTER_RHS=1 FABIPB_REUSE_MESH=1)

  if [[ "$rhs_case" == "tree" ]]; then
    env_args+=(FABIPB_FORCE_TREE_RHS=1)
  fi

  mkdir -p "$case_dir"
  ln -sf "$pqr" "$base.pqr"

  {
    printf '%q ' env "${env_args[@]}" "$repo_root/scripts/with_benchmark_env.sh" \
      "$fabipb_bin" -B=1 -g="$fmm_gpu" -m=1 "$mesh_arg" -t="$fmm_depth" \
      -eps1="$pdie" -eps2="$sdie" -P=-1 -q="$q_order" ./input
    printf '\n'
  } >"$case_dir/command.txt"

  echo "[rhs-density] $mesh_case / $rhs_case"
  (
    cd "$case_dir"
    env "${env_args[@]}" "$repo_root/scripts/with_benchmark_env.sh" \
      "$fabipb_bin" -B=1 -g="$fmm_gpu" -m=1 "$mesh_arg" -t="$fmm_depth" \
      -eps1="$pdie" -eps2="$sdie" -P=-1 -q="$q_order" ./input \
      >"$case_dir/fmm.log" 2>&1
  )
}

run_case "default_R${default_r//./p}" "-R=$default_r" "default"
run_case "default_R${default_r//./p}" "-R=$default_r" "tree"
run_case "high_density${high_density//./p}" "-d=$high_density" "default"
run_case "high_density${high_density//./p}" "-d=$high_density" "tree"

python3 - "$out_dir" <<'PY'
import csv
import re
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])

patterns = {
    "mesh_input": re.compile(
        r"Mesh input: panel=(?P<panel>\S+) mesh_atoms=(?P<mesh_atoms>\d+) "
        r"charge_atoms=(?P<charge_atoms>\d+) mode=(?P<mesh_mode>\S+) "
        r"param=(?P<mesh_param>\S+)"
    ),
    "mesh_control": re.compile(
        r"Mesh control: (?P<mesh_control_name>[^=]+)=(?P<mesh_control_value>\S+) "
        r"resolved-(?P<backend_param_name>[^=]+)=(?P<backend_param_value>\S+)"
    ),
    "raw_counts": re.compile(r"Mesh raw counts: vertices=(?P<vertices>\d+) faces=(?P<faces>\d+)"),
    "filtered": re.compile(r"#ele=(?P<input_faces>\d+), Mesh filtered panels: kept=(?P<panels>\d+) area=(?P<area>\S+)"),
    "rhs": re.compile(
        r"setupRHS direct pairs: panels=(?P<rhs_panels>\d+) charges=(?P<rhs_charges>\d+) "
        r"pairs=(?P<rhs_pairs>\d+) limit=(?P<rhs_limit>\d+) mode=(?P<rhs_mode>\S+)"
        r"(?: theta=(?P<rhs_theta>\S+))?"
    ),
    "charge_tree": re.compile(r"Charge tree build: (?P<charge_tree_time>\S+) s"),
    "stages": re.compile(r"Stage times \(s\): (?P<fields>.*)"),
}

def parse_fields(field_text):
    values = {}
    for token in field_text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values

rows = []
for log_path in sorted(out_dir.glob("*/*/fmm.log")):
    text = log_path.read_text(errors="replace")
    row = {
        "mesh_case": log_path.parent.parent.name,
        "requested_rhs": log_path.parent.name,
        "status": "ok" if "FABIPB_STOP_AFTER_RHS set" in text else "check_log",
        "log": str(log_path.relative_to(out_dir)),
    }
    for name, pattern in patterns.items():
        match = pattern.search(text)
        if not match:
            continue
        if name == "stages":
            row.update(parse_fields(match.group("fields")))
        else:
            row.update({k: v for k, v in match.groupdict().items() if v is not None})
    rows.append(row)

columns = [
    "mesh_case",
    "requested_rhs",
    "status",
    "mesh_mode",
    "mesh_control_name",
    "mesh_control_value",
    "backend_param_name",
    "backend_param_value",
    "mesh_atoms",
    "charge_atoms",
    "vertices",
    "faces",
    "panels",
    "area",
    "rhs_pairs",
    "rhs_mode",
    "rhs_theta",
    "charge_tree_time",
    "loadPanel",
    "gkInit",
    "setupFMM",
    "setupRHS",
    "log",
]

with (out_dir / "summary.csv").open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

with (out_dir / "README.md").open("w") as fh:
    fh.write("# 1a63 MSMS RHS Density Test\n\n")
    fh.write("RHS-only FABIPB runs with MSMS meshes. Each run uses ")
    fh.write("`FABIPB_STOP_AFTER_RHS=1`; the `tree` rows also use ")
    fh.write("`FABIPB_FORCE_TREE_RHS=1`.\n\n")
    fh.write("See `summary.csv` for mesh counts and stage timings.\n")

print(f"Wrote {out_dir / 'summary.csv'}")
PY

echo "[rhs-density] results: $out_dir"
