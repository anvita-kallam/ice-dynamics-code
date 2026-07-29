#!/usr/bin/env python3
"""Aggregate Totten C-sensitivity experiment summaries into a ranked report.

Expects per-experiment folders under:
  outputs/figures/vi/totten_c_sensitivity/<tag>/experiment_summary.json

  python scripts/totten_c_sensitivity_report.py
  python scripts/totten_c_sensitivity_report.py --root outputs/figures/vi/totten_c_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/figures/vi/totten_c_sensitivity"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return p.parse_args()


def _score(row: dict) -> float:
    """Higher = more distinguishable end-members (prefer low grounded corr + large Δη)."""
    corr_g = row.get("log10_eta_corr_grounded")
    deta = row.get("mean_abs_deta_grounded")
    if corr_g is None or deta is None or not np.isfinite(corr_g) or not np.isfinite(deta):
        return float("-inf")
    # Emphasize drop in grounded correlation and absolute Δη magnitude.
    return (0.97 - float(corr_g)) * 10.0 + float(deta)


def main() -> int:
    args = parse_args()
    # Resolve relative --root against cwd (sbatch runs from Archive/ with ../outputs/...).
    root = args.root.expanduser()
    root = root if root.is_absolute() else (Path.cwd() / root)
    root = root.resolve()
    rows = []
    for summary in sorted(root.glob("*/experiment_summary.json")):
        data = json.loads(summary.read_text())
        losses_no = data.get("losses", {}).get("no", {})
        losses_max = data.get("losses", {}).get("max", {})
        rows.append(
            {
                "tag": data.get("tag", summary.parent.name),
                "path": str(summary.parent),
                "log10_eta_corr_full": data.get("log10_eta_corr_full"),
                "log10_eta_corr_grounded": data.get("log10_eta_corr_grounded"),
                "log10_eta_corr_floating": data.get("log10_eta_corr_floating"),
                "mean_abs_deta_full": data.get("delta_eta", {}).get("mean_abs_full"),
                "mean_abs_deta_grounded": data.get("delta_eta", {}).get("mean_abs_grounded"),
                "max_abs_deta_grounded": data.get("delta_eta", {}).get("max_abs_grounded"),
                "phys_no": losses_no.get("train_phys"),
                "phys_max": losses_max.get("train_phys"),
                "elbo_no": losses_no.get("train_total"),
                "elbo_max": losses_max.get("train_total"),
                "shift_no": data.get("eta_log_shift", {}).get("no"),
                "shift_max": data.get("eta_log_shift", {}).get("max"),
            }
        )

    if not rows:
        print(f"No experiment_summary.json under {root}")
        print("Run totten_c_sensitivity_analyze.py after each experiment finishes.")
        return 1

    for row in rows:
        row["score"] = _score(row)
    rows.sort(key=lambda r: r["score"], reverse=True)

    # Table CSV
    import csv

    csv_path = root / "comparison_table.csv"
    fields = [
        "rank",
        "tag",
        "score",
        "log10_eta_corr_full",
        "log10_eta_corr_grounded",
        "log10_eta_corr_floating",
        "mean_abs_deta_full",
        "mean_abs_deta_grounded",
        "max_abs_deta_grounded",
        "phys_no",
        "phys_max",
        "elbo_no",
        "elbo_max",
        "shift_no",
        "shift_max",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            out = {"rank": i, **{k: row.get(k) for k in fields if k != "rank"}}
            w.writerow(out)

    # Ranking figure
    tags = [r["tag"] for r in rows]
    corr_g = [r["log10_eta_corr_grounded"] for r in rows]
    deta = [r["mean_abs_deta_grounded"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].barh(tags[::-1], corr_g[::-1], color="steelblue")
    axes[0].axvline(0.9, color="k", ls="--", lw=1, label="0.9")
    axes[0].set_xlabel("log10 η corr (grounded)")
    axes[0].set_title("Lower is more distinguishable")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[1].barh(tags[::-1], deta[::-1], color="darkorange")
    axes[1].set_xlabel("mean |Δη| grounded (MPa·yr)")
    axes[1].set_title("Higher is more distinguishable")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.suptitle("Totten C-sensitivity ranking", fontsize=12)
    fig_path = root / "ranking_overview.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Markdown report
    md = [
        "# Totten C-sensitivity comparison report",
        "",
        "Ranked by grounded η distinguishability: "
        "`score = 10*(0.97 − corr_grounded) + mean|Δη|_grounded`.",
        "",
        "SSA / Icepack equations were **not** modified; only cfg flags "
        "(physics region weights, η shift freeze, GP flexibility).",
        "",
        "| Rank | Experiment | corr (full) | corr (grounded) | corr (floating) | "
        "mean\\|Δη\\| gnd | max\\|Δη\\| gnd | phys no / max | ELBO no / max |",
        "|-----:|------------|------------:|----------------:|----------------:|"
        "---------------:|--------------:|--------------:|--------------:|",
    ]
    for i, r in enumerate(rows, start=1):
        md.append(
            f"| {i} | `{r['tag']}` | "
            f"{r['log10_eta_corr_full']:.4f} | "
            f"{r['log10_eta_corr_grounded']:.4f} | "
            f"{(r['log10_eta_corr_floating'] or float('nan')):.4f} | "
            f"{r['mean_abs_deta_grounded']:.4g} | "
            f"{r['max_abs_deta_grounded']:.4g} | "
            f"{r.get('phys_no')} / {r.get('phys_max')} | "
            f"{r.get('elbo_no')} / {r.get('elbo_max')} |"
        )
    md.extend(
        [
            "",
            "## Interpretation guide",
            "",
            "- If **corr_floating ≫ corr_grounded**, global metrics were masking C effects.",
            "- If **freeze_shift** drops correlation, the global intercept was absorbing residuals.",
            "- If **phys_w\\*** helps, floating-dominated averaging was diluting basal-friction signal.",
            "- If **gp_flex / gp_short_ls** helps, the prior was too smooth to express C structure.",
            "",
            f"Figures: `{fig_path.relative_to(ROOT) if fig_path.is_relative_to(ROOT) else fig_path}`",
            f"Table: `{csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}`",
            "",
        ]
    )
    report_path = root / "COMPARISON_REPORT.md"
    report_path.write_text("\n".join(md) + "\n")

    payload = {"ranking": rows, "score_definition": "10*(0.97-corr_g)+mean_abs_deta_g"}
    (root / "comparison_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {fig_path}")
    print("Ranking (best first):")
    for i, r in enumerate(rows, start=1):
        print(
            f"  {i}. {r['tag']}: corr_g={r['log10_eta_corr_grounded']:.4f} "
            f"mean|Δη|_g={r['mean_abs_deta_grounded']:.4g} score={r['score']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
