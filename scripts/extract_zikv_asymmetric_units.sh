#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <full-zikv.pqr> <unit-count> <output.pqr>" >&2
  exit 2
fi

input="$1"
unit_count="$2"
output="$3"

if [[ ! -f "$input" ]]; then
  echo "Input PQR is missing: $input" >&2
  exit 1
fi
if ! [[ "$unit_count" =~ ^[1-9][0-9]*$ ]] || (( unit_count > 60 )); then
  echo "unit-count must be an integer from 1 through 60" >&2
  exit 1
fi

mkdir -p "$(dirname "$output")"
awk -v requested="$unit_count" '
  /^(ATOM|HETATM)/ && $3 == "N" && $4 == "ILE" && $5 == "A" && $6 == 1 {
    unit++
    if (unit > requested) exit
  }
  unit > 0 { print }
' "$input" >"$output"

atoms="$(awk '/^(ATOM|HETATM)/{n++} END{print n+0}' "$output")"
starts="$(awk '/^(ATOM|HETATM)/ && $3=="N" && $4=="ILE" && $5=="A" && $6==1{n++} END{print n+0}' "$output")"
if [[ "$starts" != "$unit_count" || "$atoms" == "0" ]]; then
  echo "Extraction validation failed: requested=$unit_count starts=$starts atoms=$atoms" >&2
  exit 1
fi

echo "Wrote $output: units=$unit_count atoms=$atoms"
