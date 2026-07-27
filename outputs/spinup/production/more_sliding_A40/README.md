# more_sliding with A = 40

Isolated production spin-up (does **not** overwrite A=20 `more_sliding/` or A=80 attempts).

Driven by [`notebooks/spinup/run_spinup_more_sliding_A80.py`](../../../notebooks/spinup/run_spinup_more_sliding_A80.py) with `A=40` / `case_id=more_sliding_A40`.

| | Baseline | This case |
|--|--|--|
| A | 20 | **40** |
| C | 1e-3 | 1e-3 |
| Stage time | 10500 yr | 10500 yr coarse CG1; **4000 yr** later stages with C re-ramp |
| Post-handoff dt | 0.25 | **0.1** |
| Expected η | ~15 MPa·yr mean | lower (softer ice) |

Soft-ice projection handoffs re-ramp C (CG1→CG2 and coarse→fine).

## Saved config

- [`spinup_config.json`](spinup_config.json) — parameters
- [`notebooks/spinup/run_spinup_more_sliding_A80.py`](../../../notebooks/spinup/run_spinup_more_sliding_A80.py)

## Rerun (local Firedrake)

```bash
cd "/Users/anvitakallam/Ice Dynamics"
export PATH="$HOME/firedrake-env/bin:$PATH"
export PETSC_DIR="$HOME/firedrake-env"
export OMP_NUM_THREADS=1
mkdir -p outputs/logs/spinup
caffeinate -dims python -u notebooks/spinup/run_spinup_more_sliding_A80.py \
  2>&1 | tee outputs/logs/spinup/more_sliding_A40_production.log
```

Outputs when finished:
- `SteadyState_more_sliding_A40_10500yr_ramp4000_1refine.h5`
- `...json`
- `..._grid.npz`
