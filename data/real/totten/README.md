# Totten Glacier observations (2022 shelf grid)

File: `totten_vel_bed_surf_grid_shelf_2022.npz` (~4 MB)

## Contents

| Key | Shape / type | Notes |
|-----|--------------|-------|
| `h` | `(318, 174)` | ice thickness |
| `ux`, `uy` | `(318, 174)` | velocity components |
| `ux_err`, `uy_err` | `(318, 174)` | velocity uncertainties |
| `s` | `(318, 174)` | surface elevation |
| `bed` | `(318, 174)` | bed elevation |
| `bed_err`, `surf_err` | `(318, 174)` | geometry uncertainties |
| `xmin`, `xmax`, `ymin`, `ymax` | scalars | domain bounds (m) |

## Not ready for Archive VI yet

[`Archive/utilities_torch.py`](../../../Archive/utilities_torch.py) `load_snapshot` expects a spin-up-like schema. This file is missing:

- Full `x` / `y` (or `X` / `Y`) coordinate meshes
- Optional `viscosity` (needed only for η scoring against truth)
- Optional `cfg_json` for icepack physics constants (`A`, `C`, …)

**Next adapter step:** build `x`/`y` from the bounds and array shape `(318, 174)`, then map fields into the Archive snapshot loader. Do not point a sequential VI config at this NPZ until that adapter exists.
