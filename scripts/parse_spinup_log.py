"""Parse domain-mean diagnostics from spin-up terminal logs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

STAGE_ORDER = ("coarse_low", "coarse_high", "fine_low", "fine_high")

STEP_LINE_RE = re.compile(
    r"(?P<stage>coarse_low|coarse_high|fine_low|fine_high) "
    r"step (?P<step>\d+)/(?P<nsteps>\d+), "
    r"t=(?P<t>[0-9.]+) yr, C=(?P<C>[0-9.e+-]+): "
    r"avg h=(?P<avg_h>[0-9.]+), min h=(?P<min_h>[0-9.]+)"
    r"(?:, avg speed=(?P<avg_speed>[0-9.]+))?"
)

NOTEBOOK_LOG_STEMS = {
    "more_sliding": "spinupNewFull-moreSlide",
    "no_sliding": "spinupNewFull-lessSlide",
}


@dataclass(frozen=True)
class SpinupLogPoint:
    stage: str
    step: int
    nsteps: int
    t_yr: float
    C: float
    avg_h: float
    min_h: float
    avg_speed: float | None


def parse_spinup_log(path: str | Path) -> list[SpinupLogPoint]:
    """Return ordered diagnostic points printed during a spin-up run."""
    text = Path(path).read_text(errors="ignore")
    points: list[SpinupLogPoint] = []
    for line in text.splitlines():
        match = STEP_LINE_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        points.append(
            SpinupLogPoint(
                stage=groups["stage"],
                step=int(groups["step"]),
                nsteps=int(groups["nsteps"]),
                t_yr=float(groups["t"]),
                C=float(groups["C"]),
                avg_h=float(groups["avg_h"]),
                min_h=float(groups["min_h"]),
                avg_speed=(
                    float(groups["avg_speed"])
                    if groups.get("avg_speed") is not None
                    else None
                ),
            )
        )
    return points


def stage_points(points: list[SpinupLogPoint], stage: str) -> list[SpinupLogPoint]:
    return [p for p in points if p.stage == stage]


def find_latest_run_log(project_root: Path, case_id: str) -> Path | None:
    """Newest outputs/logs/spinup/<notebook>_<timestamp>/run.log for a case."""
    stem = NOTEBOOK_LOG_STEMS.get(case_id)
    if stem is None:
        return None
    log_root = project_root / "outputs" / "logs" / "spinup"
    if not log_root.is_dir():
        return None
    candidates = sorted(
        log_root.glob(f"{stem}_*/run.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def stage_local_years(points: list[SpinupLogPoint]) -> tuple[list[float], list[float]]:
    """Convert absolute spin-up time to years since the stage started."""
    if not points:
        return [], []
    t0 = points[0].t_yr
    return [p.t_yr - t0 for p in points], [p.avg_h for p in points]


def series_or_none(
    points: list[SpinupLogPoint], attr: str
) -> list[float] | None:
    values = [getattr(p, attr) for p in points]
    if not values or any(v is None for v in values):
        return None
    return values  # type: ignore[return-value]


def load_spinup_history(path: str | Path) -> list[SpinupLogPoint]:
    """Load diagnostics saved as <stem>_history.json during spin-up."""
    payload = json.loads(Path(path).read_text())
    points: list[SpinupLogPoint] = []
    for item in payload.get("points", []):
        points.append(
            SpinupLogPoint(
                stage=item["stage"],
                step=int(item["step"]),
                nsteps=int(item["nsteps"]),
                t_yr=float(item["t_yr"]),
                C=float(item["C"]),
                avg_h=float(item["avg_h"]),
                min_h=float(item["min_h"]),
                avg_speed=(
                    float(item["avg_speed"])
                    if item.get("avg_speed") is not None
                    else None
                ),
            )
        )
    return points


def resolve_spinup_diagnostics(
    project_root: Path, case_spec: dict
) -> tuple[list[SpinupLogPoint], Path | None]:
    """Prefer saved history JSON; fall back to the newest terminal run log."""
    history_path = case_spec.get("history_json")
    if history_path is not None and Path(history_path).is_file():
        return load_spinup_history(history_path), Path(history_path)

    log_path = find_latest_run_log(project_root, case_spec["case_id"])
    if log_path is not None:
        return parse_spinup_log(log_path), log_path

    return [], None
