"""Shared path layout for the Ice Dynamics project."""

from __future__ import annotations

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPINUP_DIR = OUTPUTS_DIR / "spinup"
FIGURES_DIR = OUTPUTS_DIR / "figures"
VI_DIR = OUTPUTS_DIR / "vi"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ENV_DIR = PROJECT_ROOT / "env"

# Active spin-up experiment tags (subdir + stem suffix). None = legacy default stems.
PRODUCTION_RUN_TAGS: dict[str, str | None] = {
    "more_sliding": None,
    "no_sliding": None,
}
TEST_RUN_TAGS: dict[str, str | None] = {
    "more_sliding": None,
    "no_sliding": None,
}

# Test stems (before optional RUN_TAG suffix).
TEST_STEMS: dict[str, str] = {
    "more_sliding": "SteadyState_more_sliding_test_200yr_ramp4000_1refine",
    "no_sliding": "SteadyState_no_sliding_test_200yr_ramp4000_1refine",
}


def spinup_case_dir(mode: str, case_id: str, run_tag: str | None = None) -> Path:
    path = SPINUP_DIR / mode / case_id
    if run_tag:
        path = path / run_tag
    return path


def figures_case_dir(mode: str, case_id: str, run_tag: str | None = None) -> Path:
    path = FIGURES_DIR / mode / case_id
    if run_tag:
        path = path / run_tag
    return path


def spinup_stem(mode: str, case_id: str, run_tag: str | None = None) -> str:
    if mode == "production":
        stems = {
            "more_sliding": "SteadyState_more_sliding_10500yr_ramp4000_1refine",
            "no_sliding": "SteadyState_no_sliding_10500yr_ramp4000_1refine",
        }
    elif mode == "test":
        stems = TEST_STEMS
    else:
        raise ValueError(f"Unknown mode: {mode}")
    stem = stems[case_id]
    if run_tag:
        stem = f"{stem}_{run_tag}"
    return stem


def run_tag_for(mode: str, case_id: str) -> str | None:
    tags = PRODUCTION_RUN_TAGS if mode == "production" else TEST_RUN_TAGS
    return tags.get(case_id)


def spinup_grid_npz(mode: str, case_id: str, run_tag: str | None = None) -> Path:
    tag = run_tag if run_tag is not None else run_tag_for(mode, case_id)
    return spinup_case_dir(mode, case_id, tag) / f"{spinup_stem(mode, case_id, tag)}_grid.npz"


def spinup_checkpoint(mode: str, case_id: str, run_tag: str | None = None) -> Path:
    tag = run_tag if run_tag is not None else run_tag_for(mode, case_id)
    return spinup_case_dir(mode, case_id, tag) / f"{spinup_stem(mode, case_id, tag)}.h5"


def spinup_history_json(mode: str, case_id: str, run_tag: str | None = None) -> Path:
    tag = run_tag if run_tag is not None else run_tag_for(mode, case_id)
    return spinup_case_dir(mode, case_id, tag) / f"{spinup_stem(mode, case_id, tag)}_history.json"


def build_case_spec(mode: str, case_id: str, sliding_regime: str) -> dict:
    run_tag = run_tag_for(mode, case_id)
    stem = spinup_stem(mode, case_id, run_tag)
    return {
        "case_id": case_id,
        "sliding_regime": sliding_regime,
        "run_tag": run_tag,
        "stem": stem,
        "spinup_dir": spinup_case_dir(mode, case_id, run_tag),
        "figures_dir": figures_case_dir(mode, case_id, run_tag),
        "grid_npz": spinup_grid_npz(mode, case_id, run_tag),
        "checkpoint": spinup_checkpoint(mode, case_id, run_tag),
        "history_json": spinup_history_json(mode, case_id, run_tag),
    }


PRODUCTION_CASES = {
    "more_sliding": build_case_spec("production", "more_sliding", "high"),
    "no_sliding": build_case_spec("production", "no_sliding", "low"),
}

TEST_CASES = {
    "more_sliding": build_case_spec("test", "more_sliding", "high"),
    "no_sliding": build_case_spec("test", "no_sliding", "low"),
}
