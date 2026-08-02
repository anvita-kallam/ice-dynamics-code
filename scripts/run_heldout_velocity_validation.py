#!/usr/bin/env python3
"""End-to-end held-out velocity validation for sequential VI on MISMIP A=20.

Steps:
  1) Build velocity-holdout dataset (does not touch the original NPZ).
  2) Run VI-only Stage 2 with the existing frozen PINN checkpoint.
  3) Evaluate train vs held-out velocity predictions and write figures.

Example:
  python scripts/run_heldout_velocity_validation.py
  python scripts/run_heldout_velocity_validation.py --skip-train   # eval only
  python scripts/run_heldout_velocity_validation.py --smoke        # 2-epoch dry run
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_CFG = ARCHIVE / "configs/heldout_velocity_validation/run_torch_vi_only_a20_holdout.cfg"
DEFAULT_OUT = ROOT / "outputs/heldout_velocity_validation"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Override n_epochs=2 and early_stop_patience=0 for a short run.",
    )
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _read_holdout_knobs(cfg_path: Path) -> tuple[float, int, Path, Path]:
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.optionxform = str
    cfg.read(cfg_path)
    data = cfg["data"]
    frac = float(data.get("velocity_holdout_fraction", "0.2"))
    seed = int(float(data.get("velocity_holdout_seed", "42")))
    source = Path(data.get(
        "velocity_holdout_source",
        "../outputs/spinup/production/more_sliding/"
        "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz",
    ))
    if not source.is_absolute():
        source = (ARCHIVE / source).resolve()
    holdout_npz = Path(data["h5file"])
    if not holdout_npz.is_absolute():
        holdout_npz = (ARCHIVE / holdout_npz).resolve()
    return frac, seed, source, holdout_npz


def _prepare(cfg_path: Path, out_dir: Path) -> None:
    frac, seed, source, holdout_npz = _read_holdout_knobs(cfg_path)
    dataset_dir = holdout_npz.parent
    cmd = [
        sys.executable,
        str(ROOT / "scripts/prepare_velocity_holdout_dataset.py"),
        "--source",
        str(source),
        "--output-dir",
        str(dataset_dir),
        "--holdout-fraction",
        str(frac),
        "--seed",
        str(seed),
        "--tag",
        f"f{frac:g}_s{seed}",
    ]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    # Ensure cfg-referenced paths exist (prepare writes tagged names matching cfg).
    if not holdout_npz.is_file():
        raise FileNotFoundError(f"Expected holdout NPZ at {holdout_npz}")
    out_dir.mkdir(parents=True, exist_ok=True)


def _train(cfg_path: Path, smoke: bool) -> None:
    train_cfg = cfg_path
    tmp = None
    if smoke:
        cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.optionxform = str
        cfg.read(cfg_path)
        cfg["train"]["n_epochs"] = "2"
        cfg["train"]["early_stop_patience"] = "0"
        cfg["train"]["eta_eval_every"] = "1"
        cfg["train"]["test_every"] = "1"
        cfg["train"]["plot_every"] = "1"
        cfg["train"]["checkdir"] = (
            "checkpoints/torch_vi_only/heldout_velocity_validation/a20_more_sliding_smoke"
        )
        cfg["train"]["logfile"] = "logs/log_vi_only_heldout_velocity_a20_smoke"
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".cfg",
            prefix="holdout_smoke_",
            dir=str(cfg_path.parent),
            delete=False,
        )
        cfg.write(tmp)
        tmp.close()
        train_cfg = Path(tmp.name)
        print(f"smoke cfg: {train_cfg}", flush=True)

    cmd = [sys.executable, "train_vi_only_torch.py", str(train_cfg)]
    print(">> (cwd=Archive)", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=str(ARCHIVE), check=True)
    finally:
        if tmp is not None:
            Path(tmp.name).unlink(missing_ok=True)


def _eval(cfg_path: Path, out_dir: Path, checkpoint: str, device: str | None) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts/evaluate_heldout_velocity_validation.py"),
        "--cfg",
        str(cfg_path),
        "--output-dir",
        str(out_dir),
        "--checkpoint",
        checkpoint,
    ]
    if device:
        cmd.extend(["--device", device])
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    cfg_path = args.cfg.resolve()
    out_dir = args.output_dir.resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)

    if not args.skip_prepare:
        _prepare(cfg_path, out_dir)
    if not args.skip_train:
        pretrain = ARCHIVE / "checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt"
        if not pretrain.is_file():
            raise FileNotFoundError(
                f"Missing pretrained PINN checkpoint: {pretrain}\n"
                "Reuse the existing sequential pretrain; do not retrain from scratch."
            )
        _train(cfg_path, smoke=args.smoke)
    if not args.skip_eval:
        ckpt_choice = "latest" if args.smoke else args.checkpoint
        # Smoke writes to a different checkdir; point eval at smoke cfg if needed.
        eval_cfg = cfg_path
        if args.smoke:
            # Build a temporary eval cfg that points at the smoke checkdir.
            cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
            cfg.optionxform = str
            cfg.read(cfg_path)
            cfg["train"]["checkdir"] = (
                "checkpoints/torch_vi_only/heldout_velocity_validation/a20_more_sliding_smoke"
            )
            tmp = out_dir / "smoke_eval.cfg"
            out_dir.mkdir(parents=True, exist_ok=True)
            with tmp.open("w") as f:
                cfg.write(f)
            eval_cfg = tmp
            ckpt_choice = "latest"
        _eval(eval_cfg, out_dir, ckpt_choice, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
