# more_sliding with A = 80

Isolated production spin-up (does **not** overwrite A=20 `more_sliding/`).

| | Baseline | This case |
|--|--|--|
| A | 20 | **80** |
| C | 1e-3 | 1e-3 |
| Stage time | 10500 yr | 10500 yr |
| Expected η | ~15 MPa·yr mean | lower (softer ice) |

## Saved config

- [`spinup_config.json`](spinup_config.json) — parameters
- [`notebooks/spinup/spinupNewFull-moreSlide-A80.ipynb`](../../../notebooks/spinup/spinupNewFull-moreSlide-A80.ipynb)
- [`notebooks/spinup/run_spinup_more_sliding_A80.py`](../../../notebooks/spinup/run_spinup_more_sliding_A80.py)

## Rerun (local Firedrake)

```bash
cd "/Users/anvitakallam/Ice Dynamics"
export PATH="$HOME/firedrake-env/bin:$PATH"
export PETSC_DIR="$HOME/firedrake-env"
export OMP_NUM_THREADS=1
mkdir -p outputs/logs/spinup
caffeinate -dims python -u notebooks/spinup/run_spinup_more_sliding_A80.py \
  2>&1 | tee outputs/logs/spinup/more_sliding_A80_production.log
```

Outputs when finished:
- `SteadyState_more_sliding_A80_10500yr_ramp4000_1refine.h5`
- `...json`
- `..._grid.npz`
