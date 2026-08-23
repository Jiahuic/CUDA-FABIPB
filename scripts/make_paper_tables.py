#!/usr/bin/env python3
"""Regenerate the manuscript table CSVs from results/paper/benchmarks.

Reads each benchmark directory's fmm.log plus the provenance files the run
drivers write (git_commit.txt, loadavg_before/after.txt, time_v.txt), and emits
the four tables results_plan.md section 9 specifies. Directories carrying
NOT_A_BENCHMARK.txt are skipped.

Timing rule: a run is only eligible for the timing tables if its 15-minute load
average stayed below the core count. Converged results (iterations, residual,
energy) are reported regardless, because they are unaffected by host load.
"""
import csv, os, re, glob, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "results", "paper", "benchmarks")
TABLES = os.path.join(ROOT, "results", "paper", "tables")
NCORES = os.cpu_count() or 72


def grab(text, pattern, cast=str, last=True):
    m = re.findall(pattern, text)
    if not m:
        return None
    try:
        return cast(m[-1] if last else m[0])
    except ValueError:
        return None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def scan(d):
    log = os.path.join(d, "fmm.log")
    if not os.path.isfile(log) or os.path.isfile(os.path.join(d, "NOT_A_BENCHMARK.txt")):
        return None
    t = read(log)
    if "ttl time" not in t:
        return None  # crashed or refused; not a result
    r = {
        "run": os.path.basename(d.rstrip("/")),
        "commit": (read(os.path.join(d, "git_commit.txt")) or "unrecorded")[:7],
        "panels": grab(t, r"kept=(\d+)", int),
        "vertices": grab(t, r"vertices=(\d+)", int),
        "iterations": grab(t, r"iterations=(\d+)", int),
        "residual": grab(t, r"final-residual=([0-9.eE+-]+)"),
        "energy_kcal_mol": grab(t, r"solvation energy:\s*(-?[0-9.]+)"),
        "total_s": grab(t, r"ttl time:\s*([0-9.]+)", float),
        "depth": grab(t, r"nLev=(\d+)", int, last=False),
    }
    stages = re.findall(
        r"Top-level stage times \(s\): loadPanel=([0-9.]+) gkInit=([0-9.]+) "
        r"setupFMM=([0-9.]+) setupPC=([0-9.]+) setupRHS=([0-9.]+) "
        r"gmres=([0-9.]+) treecode=([0-9.]+)", t)
    if stages:
        (r["loadPanel_s"], r["gkInit_s"], r["setupFMM_s"], r["setupPC_s"],
         r["setupRHS_s"], r["gmres_s"], r["energy_s"]) = stages[-1]
    fmm = re.findall(
        r"FMM stage totals \(s\): Q2M=([0-9.]+) M2M=([0-9.]+) M2L=([0-9.]+) "
        r"L2L=([0-9.]+) L2P=([0-9.]+) Near=([0-9.]+)", t)
    if fmm:
        (r["Q2M_s"], r["M2M_s"], r["M2L_s"], r["L2L_s"],
         r["L2P_s"], r["Near_s"]) = fmm[-1]
    # memory regime
    r["near_resident_estimate_gib"] = grab(t, r"device-cache=([0-9.]+) GiB", float)
    r["free_gib"] = grab(t, r"free=([0-9.]+) GiB", float)
    r["near_interactions"] = grab(t, r"interactions=(\d+)", int, last=False)
    r["near_regime"] = "streaming" if "GPU nearfield streaming:" in t else (
        "resident" if "GPU nearfield cache:" in t else "cpu")
    r["m2l_regime"] = "streaming" if "GPU M2L streaming cache" in t else (
        "resident" if "GPU M2L cache" in t else "cpu")
    r["auto_q2m"] = grab(t, r"Resolved GPU Q2M mode=(\d)")
    r["fallback"] = "yes" if re.search(
        r"path failed|path unavailable|panel-tree energy evaluator:", t) else "no"
    # provenance
    la = read(os.path.join(d, "loadavg_before.txt")).split()
    r["loadavg15_before"] = la[2] if len(la) > 2 else ""
    tv = read(os.path.join(d, "time_v.txt"))
    peak = grab(tv, r"Maximum resident set size \(kbytes\):\s*(\d+)", int)
    r["peak_rss_gb"] = round(peak / 1048576.0, 2) if peak else ""
    try:
        r["timing_usable"] = "yes" if float(r["loadavg15_before"]) < NCORES else "no"
    except (TypeError, ValueError):
        r["timing_usable"] = "unknown"
    return r


def write_csv(name, rows, cols):
    path = os.path.join(TABLES, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  {name}: {len(rows)} rows")


def main():
    os.makedirs(TABLES, exist_ok=True)
    rows = [r for r in (scan(d) for d in sorted(glob.glob(os.path.join(BENCH, "*/")))) if r]
    print(f"scanned {len(rows)} completed runs (cores={NCORES})")

    # --- final_benchmark: group repeats of the headline configurations -------
    groups = {
        "ZIKV_6CO8_sdens1": [r for r in rows if re.match(r"zikv_6co8_sdens1_rep\d$", r["run"])],
        "ZIKV_6CO8_sdens1.5": [r for r in rows if r["run"] == "zikv_6co8_sdens15"],
        "ZIKV_6CO8_sdens2": [r for r in rows if r["run"] == "zikv_6co8_sdens2"],
        "ZIKV_6CO8_sdens2.5": [r for r in rows if r["run"] == "zikv_6co8_sdens25"],
        "H1N1_sdens0.5": [r for r in rows if re.match(r"h1n1_sdens05_rep_\d$", r["run"])],
        "H1N1_sdens1": [r for r in rows if re.match(r"h1n1_sdens1_rep\d$", r["run"])],
        "ZIKV_6CO8_sdens1_cpu_serial": [r for r in rows if r["run"] == "zikv_6co8_sdens1_cpu_serial"],
    }
    final = []
    for case, g in groups.items():
        if not g:
            print(f"  (no runs yet for {case})")
            continue
        ts = [r["total_s"] for r in g]
        base = dict(g[0])
        base.update({
            "case": case,
            "repeat_n": len(g),
            "time_median_s": round(statistics.median(ts), 2),
            "time_min_s": round(min(ts), 2),
            "time_max_s": round(max(ts), 2),
            "time_spread_pct": round(100 * (max(ts) - min(ts)) / statistics.median(ts), 3),
        })
        final.append(base)
    write_csv("final_benchmark.csv", final, [
        "case", "commit", "panels", "iterations", "residual", "energy_kcal_mol",
        "repeat_n", "time_median_s", "time_min_s", "time_max_s", "time_spread_pct",
        "peak_rss_gb", "loadavg15_before", "timing_usable", "fallback", "run"])

    write_csv("stage_breakdown.csv", final, [
        "case", "loadPanel_s", "gkInit_s", "setupFMM_s", "setupPC_s",
        "setupRHS_s", "gmres_s", "energy_s",
        "Q2M_s", "M2M_s", "M2L_s", "L2L_s", "L2P_s", "Near_s"])

    write_csv("memory_regime.csv", final, [
        "case", "panels", "near_interactions", "near_resident_estimate_gib",
        "free_gib", "near_regime", "m2l_regime", "auto_q2m", "peak_rss_gb"])

    write_csv("all_runs.csv", rows, [
        "run", "commit", "panels", "depth", "iterations", "residual",
        "energy_kcal_mol", "total_s", "near_regime", "m2l_regime",
        "loadavg15_before", "timing_usable", "peak_rss_gb", "fallback"])


if __name__ == "__main__":
    main()
