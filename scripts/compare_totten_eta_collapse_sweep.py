#!/usr/bin/env python3
"""Compare Totten η-collapse sweep members vs baseline.

Reads metrics CSVs + final checkpoints (optional) and writes a comparison table
under outputs/figures/vi/totten_sliding_comparison/eta_collapse_sweep/.

  python scripts/compare_totten_eta_collapse_sweep.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_OUT = ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_collapse_sweep"

SWEEP = [
    {
        "name": "baseline_eta_init_15_log_clamp",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_15.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_15.csv",
        "fallback_metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_no_sliding.csv",
        "eta_maps": ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_maps.npz",
        "maps_key": "eta_no_sliding",
    },
    {
        "name": "eta_init_20_log_clamp",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_20.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_20.csv",
    },
    {
        "name": "eta_init_25_log_clamp",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_25.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_25.csv",
    },
    {
        "name": "eta_init_30_log_clamp",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_30.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_30.csv",
    },
    {
        "name": "eta_init_15_softplus_floor",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_15_softplus_floor.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_15_softplus_floor.csv",
    },
    {
        "name": "eta_init_20_softplus_floor",
        "cfg": ARCHIVE / "configs/totten/eta_collapse_sweep/run_torch_vi_only_totten_eta_init_20_softplus_floor.cfg",
        "metrics": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_eta_collapse_eta_init_20_softplus_floor.csv",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def read_csv(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array(
                [float(r[k]) if r[k] not in ("", "nan", "None") else np.nan for r in rows],
                dtype=float,
            )
        except ValueError:
            continue
    return out


def last_finite(arr, default=float("nan")):
    if arr is None:
        return default
    m = np.isfinite(arr)
    if not m.any():
        return default
    return float(arr[m][-1])


def fill_from_eta_maps(row: dict, member: dict) -> dict:
    maps = member.get("eta_maps")
    key = member.get("maps_key")
    if not maps or not key or not Path(maps).is_file():
        return row
    with np.load(maps) as z:
        eta = z[key]
        eta = eta[np.isfinite(eta) & (eta > 0)]
    row.update({
        "status": "baseline_maps",
        "mean_eta": float(np.mean(eta)),
        "median_eta": float(np.median(eta)),
        "max_over_min": float(np.max(eta) / max(np.min(eta), 1e-30)),
        "frac_at_eta_min": float(np.mean(eta <= 1.001)),
        "log10_dynamic_range": float(np.log10(np.max(eta)) - np.log10(np.min(eta))),
        "eta_std": float(np.std(eta)),
        "metrics_source": str(maps),
    })
    return row


def row_from_member(member: dict) -> dict:
    metrics_path = member["metrics"]
    data = read_csv(metrics_path)
    legacy = read_csv(member["fallback_metrics"]) if member.get("fallback_metrics") else None

    row = {
        "name": member["name"],
        "metrics_source": str(metrics_path) if data is not None else None,
        "status": "missing",
        "mean_eta": float("nan"),
        "median_eta": float("nan"),
        "max_over_min": float("nan"),
        "frac_at_eta_min": float("nan"),
        "log10_dynamic_range": float("nan"),
        "final_phys": float("nan"),
        "final_kl": float("nan"),
        "eta_std": float("nan"),
        "eta_log_shift": float("nan"),
        "frac_floor_grounded": float("nan"),
    }

    if data is not None and np.any(
            np.isfinite(data.get("eta_frac_at_floor", np.array([np.nan])))):
        row["status"] = "ok"
        row.update({
            "mean_eta": last_finite(data.get("eta_pred_mean")),
            "median_eta": last_finite(data.get("eta_median")),
            "max_over_min": last_finite(data.get("eta_max_over_min")),
            "frac_at_eta_min": last_finite(data.get("eta_frac_at_floor")),
            "log10_dynamic_range": last_finite(data.get("eta_log10_range")),
            "final_phys": last_finite(data.get("train_phys")),
            "final_kl": last_finite(data.get("train_kl")),
            "eta_std": last_finite(data.get("eta_std")),
            "eta_log_shift": last_finite(data.get("eta_log_shift")),
            "frac_floor_grounded": last_finite(data.get("eta_frac_at_floor_grounded")),
        })
        if not math.isfinite(row["log10_dynamic_range"]):
            emin = last_finite(data.get("eta_min"))
            emax = last_finite(data.get("eta_max"))
            if emin > 0 and emax > 0:
                row["log10_dynamic_range"] = math.log10(emax) - math.log10(emin)
                row["max_over_min"] = emax / emin
        return row

    if member.get("eta_maps"):
        row = fill_from_eta_maps(row, member)
    if legacy is not None:
        row["final_phys"] = last_finite(legacy.get("train_phys"))
        row["final_kl"] = last_finite(legacy.get("train_kl"))
        if row["status"] == "missing":
            row["status"] = "legacy_losses_only"
            row["metrics_source"] = str(member["fallback_metrics"])
    return row


def recommend(rows: list[dict]) -> str:
    scored = []
    for r in rows:
        if r["status"] == "missing":
            continue
        frac = r["frac_at_eta_min"]
        rng = r["log10_dynamic_range"]
        if not math.isfinite(frac):
            continue
        # Prefer low floor fraction, then larger dynamic range, then finite phys
        score = (1.0 - min(max(frac, 0.0), 1.0)) + 0.15 * (rng if math.isfinite(rng) else 0.0)
        scored.append((score, r))
    if not scored:
        return "No completed sweep members yet — submit vi_totten_eta_collapse_sweep.sbatch."
    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0][1]
    return (
        f"Recommended so far: **{best['name']}** "
        f"(frac_floor={best['frac_at_eta_min']:.3f}, "
        f"log10_range={best['log10_dynamic_range']:.3f}, "
        f"mean_η={best['mean_eta']:.3g}). "
        "Re-check C-sensitivity (no vs max sliding) on the winner before adopting."
    )


def plot_member_collapse(name: str, data: dict, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epoch = data.get("epoch")
    if epoch is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    series = (
        ("eta_frac_at_floor", "Fraction at η_min", "fraction"),
        ("eta_pred_mean", "Mean η", "MPa·yr"),
        ("eta_log_shift", "η_log_shift", "log units"),
    )
    for ax, (key, title, ylabel) in zip(axes, series):
        vals = data.get(key)
        if vals is None:
            continue
        mask = np.isfinite(vals) & np.isfinite(epoch)
        if mask.any():
            ax.plot(epoch[mask], vals[mask], color="tab:red")
        if key == "eta_frac_at_floor":
            g = data.get("eta_frac_at_floor_grounded")
            if g is not None:
                gmask = np.isfinite(g) & np.isfinite(epoch)
                if gmask.any():
                    ax.plot(epoch[gmask], g[gmask], color="tab:blue", label="grounded")
                    ax.legend(fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"η collapse — {name}", fontsize=11)
    fig.savefig(out_dir / f"collapse_{name}.png", dpi=140)
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in SWEEP:
        row = row_from_member(m)
        rows.append(row)
        data = read_csv(m["metrics"])
        if data is None and m.get("fallback_metrics"):
            data = read_csv(m["fallback_metrics"])
        if data is not None and np.any(np.isfinite(data.get("eta_frac_at_floor", np.array([np.nan])))):
            plot_member_collapse(m["name"], data, args.output_dir)

    csv_path = args.output_dir / "comparison_table.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    rec = recommend(rows)
    summary = {"rows": rows, "recommendation": rec}
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# Totten η-collapse sweep comparison", "", rec, "", "| name | status | mean η | median | max/min | frac@floor | log10 range | phys | KL |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['name']} | {r['status']} | {r['mean_eta']:.4g} | {r['median_eta']:.4g} | "
            f"{r['max_over_min']:.4g} | {r['frac_at_eta_min']:.4g} | {r['log10_dynamic_range']:.4g} | "
            f"{r['final_phys']:.4g} | {r['final_kl']:.4g} |"
        )
    (args.output_dir / "COMPARISON.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
