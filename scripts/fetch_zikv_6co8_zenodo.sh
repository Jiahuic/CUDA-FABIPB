#!/usr/bin/env sh
set -eu

record_url="https://zenodo.org/records/4568768"
archive_url="${record_url}/files/pqr.zip?download=1"
archive="TABI-PB/examples/pqr.zip"
extract_dir="TABI-PB/examples/zenodo_4568768"
source_name="ZIKV_6CO8_aa_charge_vdw_addspace.pqr"
target="test_proteins/ZIKV_6CO8_zenodo.pqr"

mkdir -p "TABI-PB/examples" "$extract_dir" "test_proteins"

if [ ! -f "$archive" ]; then
  echo "Downloading $archive_url"
  curl -L -o "$archive" "$archive_url"
fi

if [ ! -f "$extract_dir/$source_name" ]; then
  unzip -j -o "$archive" "pqr/$source_name" -d "$extract_dir"
fi

cp "$extract_dir/$source_name" "$target"

awk '
  /^(ATOM|HETATM)/ {
    atoms++;
    radius = $NF + 0;
    charge = $(NF - 1) + 0;
    total_charge += charge;
    if (radius > 0) {
      mesh_atoms++;
    } else {
      zero_radius++;
    }
  }
  END {
    printf("target=%s\n", target);
    printf("atoms=%d\n", atoms);
    printf("mesh_atoms=%d\n", mesh_atoms);
    printf("zero_radius=%d\n", zero_radius);
    printf("total_charge=%.6f\n", total_charge);
    if (atoms != 1576628 || mesh_atoms != 1558076 || zero_radius != 18552) {
      exit 1;
    }
  }
' target="$target" "$target"
