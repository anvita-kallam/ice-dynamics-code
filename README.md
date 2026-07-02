# Ice Dynamics

MISMIP+-style ice sheet spin-up, analysis, and variational-inference data prep using [icepack](https://github.com/icepack/icepack) and Firedrake.

## Layout

```
notebooks/
  spinup/       Production and test spin-up (more sliding / no sliding)
  analysis/     NPZ comparison plots, VI dataset prep
  forward/      Forward continuation from checkpoints
  learning/     icepack tutorials and VI viscosity training
scripts/        Paths, spin-up runner, log parsing, VI prep and training
docs/           SSA equation reference for icepack
outputs/        Spin-up grids, figures, VI bundles (large files gitignored)
env/            Local Firedrake conda setup (not tracked)
```

## Quick start

1. Activate your Firedrake environment (`conda activate ~/firedrake-env` or similar).
2. Run spin-up from the project root:

```bash
bash scripts/run_spinup.sh more_sliding
bash scripts/run_spinup.sh no_sliding
```

3. Analyze results in `notebooks/analysis/analyze_spinup_npz_production.ipynb` or the test notebook.

## Outputs

Large binary outputs (`*.npz`, `*.h5`, `*.png`, run logs) are excluded from git. JSON run configs and VI manifests are tracked. Regenerate grids by re-running the spin-up notebooks or `run_spinup.sh`.

## References

- SSA equations used by icepack: [docs/icepack_ssa_equations.md](docs/icepack_ssa_equations.md)
- icepack GMD paper: [Shapero et al. (2021)](https://gmd.copernicus.org/articles/14/4593/2021/)
