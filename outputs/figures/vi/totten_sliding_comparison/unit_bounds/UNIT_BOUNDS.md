# Totten η bounds — Pa·yr → MPa·yr conversion

## Where defined

| Location | Role |
|----------|------|
| `Archive/utilities_torch.py` `default_cfg` `[prior]` | Global defaults |
| Totten configs under `Archive/configs/totten/` | Per-run overrides |
| Applied in `models_torch.materialize_eta` | Clamp / softplus on η |

## Original physical range (Pa·yr)

From the initial Archive defaults (pre–icepack / MPa alignment):

| Knob | Pa·yr |
|------|------|
| `eta_min` | `1.0e3` |
| `eta_max` | `1.0e10` |
| `eta_init` (same era) | `1.0e6` |

Those figures were later treated as if they were already MPa·yr (or only partially rescaled), instead of converting Pa·yr → MPa·yr.

## Conversion

\[
1\,\mathrm{MPa\cdot yr} = 10^{6}\,\mathrm{Pa\cdot yr}
\]

\[
\eta_{\mathrm{MPa\cdot yr}} = \eta_{\mathrm{Pa\cdot yr}} / 10^{6}
\]

| Knob | Pa·yr | MPa·yr |
|------|------|--------|
| `eta_min` | `1.0e3` | `1.0e-3` |
| `eta_max` | `1.0e10` | `1.0e4` |
| `eta_init` (if converted) | `1.0e6` | `1.0` |

Defaults in `utilities_torch.py`:

```ini
eta_min = 1.0e3 / 1.0e6    # 1e-3
eta_max = 1.0e10 / 1.0e6   # 1e4
```

(Do **not** divide by `3.15576e13` — that factor converts **Pa·s** ↔ MPa·yr, not Pa·yr.)

## Totten configs

| Config | `eta_min` | `eta_max` |
|--------|-----------|-----------|
| `unit_bounds/*_{no,max}_sliding.cfg` | `1e-3` | `1e4` |
| `unit_bounds/*_floor_only.cfg` | `1e-3` | `1e6` (extra headroom) |

Both keep `eta_init = 15` and leave SSA / optimizer / other priors unchanged. With `eta_min = 1e-3`, Totten is no longer floored at `1.0`.

## Run

```bash
# re-sync corrected configs, then cancel the old wrong-factor array if still running:
# scancel 1231611
cd ~/ice-dynamics/Archive
sbatch slurm/vi_totten_unit_bounds.sbatch
```
