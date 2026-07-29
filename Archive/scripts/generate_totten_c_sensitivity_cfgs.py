#!/usr/bin/env python3
"""Generate Totten C-sensitivity experiment configs from the baseline pair.

Writes under Archive/configs/totten/c_sensitivity/<experiment>/
without modifying SSA physics — only train/prior knobs behind flags.

  cd Archive
  python scripts/generate_totten_c_sensitivity_cfgs.py
"""

from __future__ import annotations

from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]
BASE_NO = ARCHIVE / "configs/totten/run_torch_vi_only_totten_no_sliding.cfg"
BASE_MAX = ARCHIVE / "configs/totten/run_torch_vi_only_totten_max_sliding.cfg"
OUT_ROOT = ARCHIVE / "configs/totten/c_sensitivity"

# (experiment_id, description, overrides applied to both no/max)
EXPERIMENTS: list[tuple[str, str, dict[str, str]]] = [
    (
        "baseline",
        "Unit-corrected baseline (phys weights 1/1, learnable η shift)",
        {
            "grounded_phys_weight": "1.0",
            "floating_phys_weight": "1.0",
            "learn_eta_shift": "True",
            "freeze_eta_log_shift": "False",
        },
    ),
    (
        "phys_w2",
        "Exp1: grounded_phys_weight=2, floating=1",
        {"grounded_phys_weight": "2.0", "floating_phys_weight": "1.0"},
    ),
    (
        "phys_w5",
        "Exp1: grounded_phys_weight=5, floating=1",
        {"grounded_phys_weight": "5.0", "floating_phys_weight": "1.0"},
    ),
    (
        "phys_w10",
        "Exp1: grounded_phys_weight=10, floating=1",
        {"grounded_phys_weight": "10.0", "floating_phys_weight": "1.0"},
    ),
    (
        "freeze_shift",
        "Exp3: freeze global eta_log_shift at 0",
        {"learn_eta_shift": "False", "freeze_eta_log_shift": "True"},
    ),
    (
        "gp_short_ls",
        "Exp4: shorter length scale (5 km), fixed LS",
        {
            "l_scale_eta": "5.0e3",
            "learnable_length_scale": "False",
        },
    ),
    (
        "gp_flex",
        "Exp4: shorter LS + more inducing + weaker KL",
        {
            "l_scale_eta": "3.0e3",
            "learnable_length_scale": "False",
            "num_inducing_x": "40",
            "num_inducing_y": "40",
            "std_eta": "3.0",
            "kl_eta": "0.05",
        },
    ),
]


def _set_key(text: str, key: str, value: str) -> str:
    """Replace first assignment of ``key = ...`` or append under [train]/[prior]."""
    lines = text.splitlines(True)
    out: list[str] = []
    replaced = False
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith(f"{key} ="):
            indent = line[: len(line) - len(stripped)]
            out.append(f"{indent}{key} = {value}\n")
            replaced = True
        else:
            out.append(line)
    if replaced:
        return "".join(out)
    # Append before next section after [train] or [prior] depending on key.
    prior_keys = {
        "l_scale_eta",
        "learnable_length_scale",
        "num_inducing_x",
        "num_inducing_y",
        "std_eta",
        "kl_eta",
    }
    section = "[prior]" if key in prior_keys else "[train]"
    out2: list[str] = []
    inserted = False
    for i, line in enumerate(out):
        out2.append(line)
        if not inserted and line.strip() == section:
            # insert after section header + any following blank/comment block's first keys area:
            # find end of consecutive non-section lines starting next
            out2.append(f"{key} = {value}\n")
            inserted = True
    if not inserted:
        out2.append(f"\n{section}\n{key} = {value}\n")
    return "".join(out2)


def _retarget(text: str, experiment: str, case: str, friction_c: float) -> str:
    text = _set_key(text, "friction_C", f"{friction_c:g}")
    text = _set_key(
        text,
        "checkdir",
        f"'checkpoints/torch_vi_only/totten/c_sensitivity/{experiment}/{case}'",
    )
    text = _set_key(
        text,
        "logfile",
        f"'logs/log_vi_only_totten_c_sens_{experiment}_{case}'",
    )
    text = _set_key(
        text,
        "evaldir",
        f"'outputs/totten/c_sensitivity/{experiment}/{case}'",
    )
    # predict.output_file and checkdir may appear twice ([train] vs [pretrain]);
    # only rewrite predict block output_file via unique path.
    text = text.replace(
        "output_file = 'outputs/totten/no_sliding/posterior_samples.h5'",
        f"output_file = 'outputs/totten/c_sensitivity/{experiment}/{case}/posterior_samples.h5'",
    )
    text = text.replace(
        "output_file = 'outputs/totten/max_sliding/posterior_samples.h5'",
        f"output_file = 'outputs/totten/c_sensitivity/{experiment}/{case}/posterior_samples.h5'",
    )
    return text


def write_pair(experiment: str, description: str, overrides: dict[str, str]) -> None:
    out_dir = OUT_ROOT / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.txt").write_text(description.strip() + "\n")
    for case, base, c_val in (
        ("no_sliding", BASE_NO, 100.0),
        ("max_sliding", BASE_MAX, 0.001),
    ):
        text = base.read_text()
        text = _retarget(text, experiment, case, c_val)
        for key, value in overrides.items():
            text = _set_key(text, key, value)
        # Ensure baseline flags exist even if source lacked them.
        for key, value in (
            ("grounded_phys_weight", overrides.get("grounded_phys_weight", "1.0")),
            ("floating_phys_weight", overrides.get("floating_phys_weight", "1.0")),
            ("learn_eta_shift", overrides.get("learn_eta_shift", "True")),
            ("freeze_eta_log_shift", overrides.get("freeze_eta_log_shift", "False")),
        ):
            text = _set_key(text, key, value)
        path = out_dir / f"run_torch_vi_only_totten_{case}.cfg"
        path.write_text(text)
        print(f"wrote {path.relative_to(ARCHIVE)}")


def main() -> int:
    if not BASE_NO.is_file() or not BASE_MAX.is_file():
        raise FileNotFoundError("Baseline Totten VI configs missing")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for exp_id, desc, overrides in EXPERIMENTS:
        write_pair(exp_id, desc, overrides)
    index = OUT_ROOT / "EXPERIMENTS.md"
    lines = [
        "# Totten C-sensitivity experiments",
        "",
        "Generated by `scripts/generate_totten_c_sensitivity_cfgs.py`.",
        "SSA / Icepack physics unchanged — only cfg flags.",
        "",
        "| ID | Description |",
        "|----|-------------|",
    ]
    for exp_id, desc, _ in EXPERIMENTS:
        lines.append(f"| `{exp_id}` | {desc} |")
    lines.extend(
        [
            "",
            "## Submit one experiment (no + max)",
            "",
            "```bash",
            "cd Archive",
            "EXP=phys_w5",
            "NO=$(sbatch --parsable slurm/vi_train_vi_only_cfg.sbatch \\",
            "  configs/totten/c_sensitivity/${EXP}/run_torch_vi_only_totten_no_sliding.cfg)",
            "MAX=$(sbatch --parsable slurm/vi_train_vi_only_cfg.sbatch \\",
            "  configs/totten/c_sensitivity/${EXP}/run_torch_vi_only_totten_max_sliding.cfg)",
            "sbatch --dependency=afterok:${NO}:${MAX} \\",
            "  slurm/vi_eval_totten_c_sensitivity.sbatch ${EXP}",
            "```",
            "",
        ]
    )
    index.write_text("\n".join(lines))
    print(f"wrote {index.relative_to(ARCHIVE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
