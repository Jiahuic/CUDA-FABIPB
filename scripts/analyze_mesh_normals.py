#!/usr/bin/env python3
"""Analyze NanoShaper/MSMS face and vertex normal consistency."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    length = norm(a)
    if length <= 0.0:
        return 0.0, 0.0, 0.0
    return a[0] / length, a[1] / length, a[2] / length


def add3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
    wa: float,
    wb: float,
    wc: float,
) -> tuple[float, float, float]:
    return (
        wa * a[0] + wb * b[0] + wc * c[0],
        wa * a[1] + wb * b[1] + wc * c[1],
        wa * a[2] + wb * b[2] + wc * c[2],
    )


def read_counted_rows(path: Path) -> tuple[int, list[list[str]]]:
    rows: list[list[str]] = []
    with path.open() as fp:
        fp.readline()
        fp.readline()
        count = int(fp.readline().split()[0])
        for line in fp:
            line = line.strip()
            if line:
                rows.append(line.split())
    return count, rows


def read_vertices(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    count, rows = read_counted_rows(path)
    xyz: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    for row in rows:
        if len(row) < 6:
            continue
        xyz.append((float(row[0]), float(row[1]), float(row[2])))
        normals.append(normalize((float(row[3]), float(row[4]), float(row[5]))))
    if len(xyz) != count:
        raise ValueError(f"{path}: expected {count} vertices, read {len(xyz)}")
    return xyz, normals


def read_faces(path: Path) -> list[tuple[int, int, int]]:
    count, rows = read_counted_rows(path)
    faces: list[tuple[int, int, int]] = []
    for row in rows:
        if len(row) < 3:
            continue
        faces.append((int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1))
    if len(faces) != count:
        raise ValueError(f"{path}: expected {count} faces, read {len(faces)}")
    return faces


def read_rhs_sample(path: Path | None) -> dict[int, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            face_key = row.get("mesh_face_idx") or row["idx"]
            rows[int(face_key)] = row
    return rows


class Stat:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.total_area = 0.0
        self.sum = 0.0
        self.area_sum = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf
        self.count_lt_0 = 0
        self.area_lt_0 = 0.0
        self.thresholds = [0.5, 0.7, 0.9, 0.99]
        self.count_lt = {threshold: 0 for threshold in self.thresholds}
        self.area_lt = {threshold: 0.0 for threshold in self.thresholds}

    def add(self, value: float, area: float) -> None:
        self.count += 1
        self.total_area += area
        self.sum += value
        self.area_sum += area * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        if value < 0.0:
            self.count_lt_0 += 1
            self.area_lt_0 += area
        for threshold in self.thresholds:
            if value < threshold:
                self.count_lt[threshold] += 1
                self.area_lt[threshold] += area

    def row(self) -> dict[str, float | int | str]:
        return {
            "metric": self.name,
            "count": self.count,
            "total_area": self.total_area,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.sum / self.count if self.count else 0.0,
            "area_weighted_mean": self.area_sum / self.total_area if self.total_area else 0.0,
            "count_lt_0": self.count_lt_0,
            "area_lt_0": self.area_lt_0,
            "count_lt_0p5": self.count_lt[0.5],
            "area_lt_0p5": self.area_lt[0.5],
            "count_lt_0p7": self.count_lt[0.7],
            "area_lt_0p7": self.area_lt[0.7],
            "count_lt_0p9": self.count_lt[0.9],
            "area_lt_0p9": self.area_lt[0.9],
            "count_lt_0p99": self.count_lt[0.99],
            "area_lt_0p99": self.area_lt[0.99],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vert", required=True)
    parser.add_argument("--face", required=True)
    parser.add_argument("--rhs-sample")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices, vertex_normals = read_vertices(Path(args.vert))
    faces = read_faces(Path(args.face))
    rhs_sample = read_rhs_sample(Path(args.rhs_sample) if args.rhs_sample else None)

    metrics = [
        "dot_geom_fabipb_default",
        "abs_dot_geom_fabipb_default",
        "dot_geom_vertex_mean",
        "dot_geometric_aligned_vertex_mean",
        "min_dot_geometric_aligned_to_face_vertices",
        "dot_fabipb_default_vertex_mean",
        "dot_fabipb_default_v0",
        "dot_fabipb_default_v1",
        "dot_fabipb_default_v2",
        "min_dot_fabipb_default_to_face_vertices",
        "min_pairwise_vertex_dot",
    ]
    stats = {name: Stat(name) for name in metrics}
    worst: list[dict[str, float | int]] = []
    sample_rows: list[dict[str, float | int | str]] = []

    for face_idx, (i0, i1, i2) in enumerate(faces):
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        n0, n1, n2 = vertex_normals[i0], vertex_normals[i1], vertex_normals[i2]
        raw_cross = cross(sub(v1, v0), sub(v2, v0))
        area = 0.5 * norm(raw_cross)
        geom = normalize(raw_cross)
        fabipb_default = normalize(add3(n0, n1, n2, 0.5, 0.25, 0.25))
        vertex_mean = normalize(add3(n0, n1, n2, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
        geometric_aligned = geom
        if dot(geometric_aligned, fabipb_default) < 0.0:
            geometric_aligned = (
                -geometric_aligned[0],
                -geometric_aligned[1],
                -geometric_aligned[2],
            )

        d_geom_default = dot(geom, fabipb_default)
        d_geom_mean = dot(geom, vertex_mean)
        d_geometric_aligned_mean = dot(geometric_aligned, vertex_mean)
        min_geometric_aligned_vertex = min(
            dot(geometric_aligned, n0),
            dot(geometric_aligned, n1),
            dot(geometric_aligned, n2),
        )
        d_default_mean = dot(fabipb_default, vertex_mean)
        d_default_v0 = dot(fabipb_default, n0)
        d_default_v1 = dot(fabipb_default, n1)
        d_default_v2 = dot(fabipb_default, n2)
        min_default_vertex = min(d_default_v0, d_default_v1, d_default_v2)
        min_pairwise_vertex = min(dot(n0, n1), dot(n0, n2), dot(n1, n2))

        values = {
            "dot_geom_fabipb_default": d_geom_default,
            "abs_dot_geom_fabipb_default": abs(d_geom_default),
            "dot_geom_vertex_mean": d_geom_mean,
            "dot_geometric_aligned_vertex_mean": d_geometric_aligned_mean,
            "min_dot_geometric_aligned_to_face_vertices": min_geometric_aligned_vertex,
            "dot_fabipb_default_vertex_mean": d_default_mean,
            "dot_fabipb_default_v0": d_default_v0,
            "dot_fabipb_default_v1": d_default_v1,
            "dot_fabipb_default_v2": d_default_v2,
            "min_dot_fabipb_default_to_face_vertices": min_default_vertex,
            "min_pairwise_vertex_dot": min_pairwise_vertex,
        }
        for name, value in values.items():
            stats[name].add(value, area)

        centroid = (
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        )
        worst_score = min(abs(d_geom_default), min_default_vertex, min_pairwise_vertex)
        worst.append({
            "face_idx": face_idx,
            "area": area,
            "centroid_x": centroid[0],
            "centroid_y": centroid[1],
            "centroid_z": centroid[2],
            "worst_score": worst_score,
            **values,
        })

        if face_idx in rhs_sample:
            row = rhs_sample[face_idx]
            rhs_normal = normalize((float(row["nx"]), float(row["ny"]), float(row["nz"])))
            sample_rows.append({
                "face_idx": face_idx,
                "vector_idx": row.get("idx", ""),
                "area": area,
                "dot_rhs_sample_fabipb_default": dot(rhs_normal, fabipb_default),
                "dot_rhs_sample_geom": dot(rhs_normal, geom),
                "dot_rhs_sample_vertex_mean": dot(rhs_normal, vertex_mean),
                "dot_geom_fabipb_default": d_geom_default,
                "dot_fabipb_default_vertex_mean": d_default_mean,
                "min_dot_fabipb_default_to_face_vertices": min_default_vertex,
                "tabi_area_avg1": row.get("tabi_area_avg1", ""),
                "integrated1": row.get("integrated1", ""),
            })

    summary_path = out_dir / "normal_summary.csv"
    summary_fields = list(next(iter(stats.values())).row().keys())
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(stat.row() for stat in stats.values())

    worst_path = out_dir / "normal_worst_faces.csv"
    worst.sort(key=lambda row: row["worst_score"])
    worst_fields = list(worst[0].keys()) if worst else []
    with worst_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=worst_fields)
        writer.writeheader()
        writer.writerows(worst[: args.top_n])

    if sample_rows:
        sample_path = out_dir / "normal_rhs_sample_join.csv"
        sample_fields = list(sample_rows[0].keys())
        with sample_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=sample_fields)
            writer.writeheader()
            writer.writerows(sample_rows)
        print(f"Wrote {sample_path}")

    print(f"Wrote {summary_path}")
    print(f"Wrote {worst_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
