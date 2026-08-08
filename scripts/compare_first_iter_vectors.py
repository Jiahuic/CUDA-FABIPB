#!/usr/bin/env python3
"""Compare first-iteration GMRES vector dumps from FABIPB and TABI-PB."""

from __future__ import annotations

import argparse
import csv
import math
from itertools import zip_longest
from pathlib import Path


DEFAULT_VECTORS = [
    "b",
    "x0",
    "r0",
    "Minv_r0",
    "v1",
    "Av1",
    "Minv_Av1",
    "x_after_iter1",
]


def vector_path(prefix: str, tag: str) -> Path:
    return Path(f"{prefix}_{tag}.csv")


def read_row(row: dict[str, str]) -> tuple[int, int, float]:
    return int(row["idx"]), int(row["component"]), float(row["value"])


def vector_stats(path: Path) -> dict[str, float]:
    count = 0
    l1 = 0.0
    l2 = 0.0
    inf_norm = 0.0
    total = 0.0
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            value = float(row["value"])
            av = abs(value)
            count += 1
            l1 += av
            l2 += value * value
            total += value
            inf_norm = max(inf_norm, av)
    return {
        "rows": count,
        "l1": l1,
        "l2": math.sqrt(l2),
        "inf": inf_norm,
        "mean": total / count if count else 0.0,
    }


def parse_metadata(path: Path) -> dict[str, dict[str, float | int | str]]:
    rows: dict[str, dict[str, float | int | str]] = {}
    if not path.exists():
        return rows
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            fields: dict[str, float | int | str] = {}
            tag = ""
            for token in line.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                if key == "tag":
                    tag = value
                    fields[key] = value
                    continue
                try:
                    if key in {"iter", "n", "stride"} or key.endswith("_n"):
                        fields[key] = int(value)
                    else:
                        fields[key] = float(value)
                except ValueError:
                    fields[key] = value
            if tag:
                rows[tag] = fields
    return rows


def read_rhs_summary(path: Path) -> dict[tuple[str, int], dict[str, float | int | str]]:
    rows: dict[tuple[str, int], dict[str, float | int | str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            quantity = row.get("quantity", "")
            component_text = row.get("component", "")
            if not quantity or component_text == "":
                continue
            component = int(component_text)
            parsed: dict[str, float | int | str] = {
                "quantity": quantity,
                "component": component,
            }
            for key, value in row.items():
                if key in {"quantity", "component"}:
                    continue
                if value == "":
                    parsed[key] = ""
                    continue
                try:
                    parsed[key] = int(value) if key == "count" else float(value)
                except ValueError:
                    parsed[key] = value
            rows[(quantity, component)] = parsed
    return rows


def metadata_comparison_rows(
    left_prefix: str,
    right_prefix: str,
    vectors: list[str],
    left_label: str,
    right_label: str,
) -> list[dict[str, object]]:
    left_meta = parse_metadata(Path(f"{left_prefix}_metadata.txt"))
    right_meta = parse_metadata(Path(f"{right_prefix}_metadata.txt"))
    tags = list(vectors)
    if "first_residual" not in tags:
        tags.append("first_residual")

    rows: list[dict[str, object]] = []
    for tag in tags:
        left = left_meta.get(tag, {})
        right = right_meta.get(tag, {})
        left_n = left.get("n", "")
        right_n = right.get("n", "")
        left_l2 = left.get("l2", "")
        right_l2 = right.get("l2", "")
        left_rms = ""
        right_rms = ""
        if isinstance(left_n, int) and left_n > 0 and isinstance(left_l2, float):
            left_rms = left_l2 / math.sqrt(left_n)
        if isinstance(right_n, int) and right_n > 0 and isinstance(right_l2, float):
            right_rms = right_l2 / math.sqrt(right_n)

        rows.append({
            "vector": tag,
            f"{left_label}_n": left_n,
            f"{right_label}_n": right_n,
            "dimension_ratio_left_over_right": (
                left_n / right_n
                if isinstance(left_n, int) and isinstance(right_n, int) and right_n > 0
                else ""
            ),
            f"{left_label}_l1": left.get("l1", ""),
            f"{right_label}_l1": right.get("l1", ""),
            f"{left_label}_l2": left_l2,
            f"{right_label}_l2": right_l2,
            f"{left_label}_rms": left_rms,
            f"{right_label}_rms": right_rms,
            f"{left_label}_inf": left.get("inf", ""),
            f"{right_label}_inf": right.get("inf", ""),
            f"{left_label}_sum": left.get("sum", ""),
            f"{right_label}_sum": right.get("sum", ""),
            f"{left_label}_mean": left.get("mean", ""),
            f"{right_label}_mean": right.get("mean", ""),
            f"{left_label}_component0_sum": left.get("component0_sum", ""),
            f"{right_label}_component0_sum": right.get("component0_sum", ""),
            f"{left_label}_component0_n": left.get("component0_n", ""),
            f"{right_label}_component0_n": right.get("component0_n", ""),
            f"{left_label}_component0_mean": left.get("component0_mean", ""),
            f"{right_label}_component0_mean": right.get("component0_mean", ""),
            f"{left_label}_component0_l2": left.get("component0_l2", ""),
            f"{right_label}_component0_l2": right.get("component0_l2", ""),
            f"{left_label}_component0_inf": left.get("component0_inf", ""),
            f"{right_label}_component0_inf": right.get("component0_inf", ""),
            f"{left_label}_component1_sum": left.get("component1_sum", ""),
            f"{right_label}_component1_sum": right.get("component1_sum", ""),
            f"{left_label}_component1_n": left.get("component1_n", ""),
            f"{right_label}_component1_n": right.get("component1_n", ""),
            f"{left_label}_component1_mean": left.get("component1_mean", ""),
            f"{right_label}_component1_mean": right.get("component1_mean", ""),
            f"{left_label}_component1_l2": left.get("component1_l2", ""),
            f"{right_label}_component1_l2": right.get("component1_l2", ""),
            f"{left_label}_component1_inf": left.get("component1_inf", ""),
            f"{right_label}_component1_inf": right.get("component1_inf", ""),
            f"{left_label}_value": left.get("value", ""),
            f"{right_label}_value": right.get("value", ""),
        })
    return rows


def rhs_sum_comparison_rows(
    left_rhs_summary: Path,
    right_prefix: str,
    left_label: str,
    right_label: str,
) -> list[dict[str, object]]:
    rhs_rows = read_rhs_summary(left_rhs_summary)
    right_meta = parse_metadata(Path(f"{right_prefix}_metadata.txt"))
    right_b = right_meta.get("b", {})
    rows: list[dict[str, object]] = []

    for component in (0, 1):
        integrated = rhs_rows.get(("integrated", component), {})
        tabi_avg = rhs_rows.get(("tabi_area_avg", component), {})
        right_n = right_b.get(f"component{component}_n", "")
        right_sum = right_b.get(f"component{component}_sum", "")
        right_mean = right_b.get(f"component{component}_mean", "")
        avg_sum = tabi_avg.get("sum", "")
        avg_mean = tabi_avg.get("mean", "")
        integrated_sum = integrated.get("sum", "")
        scaled_avg_sum = ""

        diff_avg = ""
        ratio_avg = ""
        if isinstance(avg_sum, float) and isinstance(right_sum, float):
            diff_avg = avg_sum - right_sum
            ratio_avg = avg_sum / right_sum if right_sum != 0.0 else ""

        mean_diff = ""
        mean_ratio = ""
        if isinstance(avg_mean, float) and isinstance(right_mean, float):
            mean_diff = avg_mean - right_mean
            mean_ratio = avg_mean / right_mean if right_mean != 0.0 else ""

        scaled_diff = ""
        scaled_ratio = ""
        if isinstance(avg_mean, float) and isinstance(right_n, int):
            scaled_avg_sum = avg_mean * right_n
            if isinstance(right_sum, float):
                scaled_diff = scaled_avg_sum - right_sum
                scaled_ratio = scaled_avg_sum / right_sum if right_sum != 0.0 else ""

        diff_integrated = ""
        ratio_integrated = ""
        if isinstance(integrated_sum, float) and isinstance(right_sum, float):
            diff_integrated = integrated_sum - right_sum
            ratio_integrated = integrated_sum / right_sum if right_sum != 0.0 else ""

        rows.append({
            "component": component,
            f"{right_label}_b_n": right_n,
            f"{right_label}_b_sum": right_sum,
            f"{right_label}_b_mean": right_mean,
            f"{right_label}_b_l2": right_b.get(f"component{component}_l2", ""),
            f"{right_label}_b_inf": right_b.get(f"component{component}_inf", ""),
            f"{left_label}_target_n": tabi_avg.get("count", ""),
            f"{left_label}_integrated_sum": integrated_sum,
            f"{left_label}_integrated_mean": integrated.get("mean", ""),
            f"{left_label}_integrated_l2": integrated.get("l2", ""),
            f"{left_label}_integrated_inf": integrated.get("inf", ""),
            f"{left_label}_tabi_area_avg_sum": avg_sum,
            f"{left_label}_tabi_area_avg_mean": avg_mean,
            f"{left_label}_tabi_area_avg_l2": tabi_avg.get("l2", ""),
            f"{left_label}_tabi_area_avg_inf": tabi_avg.get("inf", ""),
            f"{left_label}_tabi_area_avg_sum_scaled_to_{right_label}_n": scaled_avg_sum,
            f"{left_label}_surface_integral_from_area_avg": tabi_avg.get("area_weighted_sum", ""),
            f"{left_label}_surface_mean_from_area_avg": tabi_avg.get("area_weighted_mean", ""),
            "diff_tabi_area_avg_sum_minus_tabi_b_sum": diff_avg,
            "ratio_tabi_area_avg_sum_over_tabi_b_sum": ratio_avg,
            "diff_tabi_area_avg_mean_minus_tabi_b_mean": mean_diff,
            "ratio_tabi_area_avg_mean_over_tabi_b_mean": mean_ratio,
            "diff_count_scaled_tabi_area_avg_sum_minus_tabi_b_sum": scaled_diff,
            "ratio_count_scaled_tabi_area_avg_sum_over_tabi_b_sum": scaled_ratio,
            "diff_integrated_sum_minus_tabi_b_sum": diff_integrated,
            "ratio_integrated_sum_over_tabi_b_sum": ratio_integrated,
        })
    return rows


def compare_vector(left: Path, right: Path, top_n: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not left.exists() or not right.exists():
        return {
            "status": "missing",
            "left_rows": 0,
            "right_rows": 0,
            "max_abs": "",
            "rel_l2": "",
            "max_idx": "",
            "max_component": "",
            "left_value": "",
            "right_value": "",
            "left_l2": "",
            "right_l2": "",
        }, []

    left_stats = vector_stats(left)
    right_stats = vector_stats(right)
    if left_stats["rows"] != right_stats["rows"]:
        return {
            "status": "shape_mismatch",
            "left_rows": left_stats["rows"],
            "right_rows": right_stats["rows"],
            "max_abs": "",
            "rel_l2": "",
            "max_idx": "",
            "max_component": "",
            "left_value": "",
            "right_value": "",
            "left_l2": left_stats["l2"],
            "right_l2": right_stats["l2"],
        }, []

    max_abs = -1.0
    max_idx = -1
    max_component = -1
    max_left = 0.0
    max_right = 0.0
    l2_diff = 0.0
    l2_ref = 0.0
    top: list[dict[str, object]] = []
    status = "ok"

    with left.open(newline="") as left_fp, right.open(newline="") as right_fp:
        left_reader = csv.DictReader(left_fp)
        right_reader = csv.DictReader(right_fp)
        for left_row, right_row in zip_longest(left_reader, right_reader):
            if left_row is None or right_row is None:
                status = "shape_mismatch"
                break
            li, lc, lv = read_row(left_row)
            ri, rc, rv = read_row(right_row)
            if li != ri or lc != rc:
                status = "index_mismatch"
                break
            diff = abs(lv - rv)
            l2_diff += diff * diff
            l2_ref += lv * lv
            if diff > max_abs:
                max_abs = diff
                max_idx = li
                max_component = lc
                max_left = lv
                max_right = rv
            top.append({
                "idx": li,
                "component": lc,
                "abs_diff": diff,
                "left_value": lv,
                "right_value": rv,
            })
            top.sort(key=lambda item: item["abs_diff"], reverse=True)
            del top[top_n:]

    return {
        "status": status,
        "left_rows": left_stats["rows"],
        "right_rows": right_stats["rows"],
        "max_abs": max_abs if max_abs >= 0.0 else "",
        "rel_l2": math.sqrt(l2_diff / l2_ref) if l2_ref > 0.0 else 0.0,
        "max_idx": max_idx if max_idx >= 0 else "",
        "max_component": max_component if max_component >= 0 else "",
        "left_value": max_left,
        "right_value": max_right,
        "left_l2": left_stats["l2"],
        "right_l2": right_stats["l2"],
    }, top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-prefix", required=True, help="Left dump prefix, e.g. fmm/gmres")
    parser.add_argument("--right-prefix", required=True, help="Right dump prefix, e.g. tabi/gmres")
    parser.add_argument("--left-label", default="fmm")
    parser.add_argument("--right-label", default="tabi")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--vectors", nargs="*", default=DEFAULT_VECTORS)
    parser.add_argument(
        "--left-rhs-summary",
        help="Optional FABIPB RHS summary CSV with integrated and TABI-style area-averaged rows",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    top_path = out_dir / "top_differences.csv"
    metadata_summary_path = out_dir / "metadata_summary.csv"
    rhs_sum_comparison_path = out_dir / "rhs_sum_comparison.csv"
    rows = []
    top_rows = []

    for tag in args.vectors:
        summary, top = compare_vector(vector_path(args.left_prefix, tag),
                                      vector_path(args.right_prefix, tag),
                                      args.top_n)
        summary["vector"] = tag
        rows.append(summary)
        for rank, item in enumerate(top, 1):
            item = dict(item)
            item["vector"] = tag
            item["rank"] = rank
            top_rows.append(item)

    summary_fields = [
        "vector",
        "status",
        "left_rows",
        "right_rows",
        "max_abs",
        "rel_l2",
        "max_idx",
        "max_component",
        "left_value",
        "right_value",
        "left_l2",
        "right_l2",
    ]
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(rows)

    top_fields = ["vector", "rank", "idx", "component", "abs_diff", "left_value", "right_value"]
    with top_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=top_fields)
        writer.writeheader()
        writer.writerows(top_rows)

    metadata_rows = metadata_comparison_rows(
        args.left_prefix,
        args.right_prefix,
        list(args.vectors),
        args.left_label,
        args.right_label,
    )
    if metadata_rows:
        metadata_fields = list(metadata_rows[0].keys())
        with metadata_summary_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=metadata_fields)
            writer.writeheader()
            writer.writerows(metadata_rows)

    if args.left_rhs_summary:
        rhs_rows = rhs_sum_comparison_rows(
            Path(args.left_rhs_summary),
            args.right_prefix,
            args.left_label,
            args.right_label,
        )
        if rhs_rows:
            rhs_fields = list(rhs_rows[0].keys())
            with rhs_sum_comparison_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=rhs_fields)
                writer.writeheader()
                writer.writerows(rhs_rows)

    readme = out_dir / "README.md"
    with readme.open("w") as fp:
        fp.write("# First-Iteration Vector Comparison\n\n")
        fp.write(f"Left ({args.left_label}) prefix: `{args.left_prefix}`\n\n")
        fp.write(f"Right ({args.right_label}) prefix: `{args.right_prefix}`\n\n")
        fp.write("`shape_mismatch` is expected when comparing native TABI-PB nodepatch vectors ")
        fp.write("with native FABIPB panel vectors; in that case the summary records norm-only ")
        fp.write("information and the operator/RHS diagnostics should be used for formula-level checks.\n\n")
        fp.write("- `summary.csv`: per-vector status and norms\n")
        fp.write("- `metadata_summary.csv`: full-vector metadata norms and first residuals\n")
        if args.left_rhs_summary:
            fp.write("- `rhs_sum_comparison.csv`: TABI `b` sums vs FABIPB integrated and area-normalized RHS sums\n")
        fp.write("- `top_differences.csv`: largest per-index differences when vector shapes align\n")

    print(f"Wrote {summary_path}")
    print(f"Wrote {metadata_summary_path}")
    if args.left_rhs_summary:
        print(f"Wrote {rhs_sum_comparison_path}")
    print(f"Wrote {top_path}")
    print(f"Wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
