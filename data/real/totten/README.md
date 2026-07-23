# Totten Glacier observations (2022 shelf grid)

## Files

| File | Role |
|------|------|
| `totten_vel_bed_surf_grid_shelf_2022.npz` | Raw shelf-grid observations (~4 MB) |
| `totten_archive_vi_2022.npz` | Archive-ready product for sequential VI |
| `totten_archive_vi_2022_summary.json` | Finite-fraction / speed / packaging QC summary |

Rebuild the archive product:

```bash
python scripts/process_totten_for_archive_vi.py
```

## Raw contents

| Key | Shape / type | Notes |
|-----|--------------|-------|
| `h` | `(318, 174)` | ice thickness |
| `ux`, `uy` | `(318, 174)` | velocity components (m/yr) |
| `ux_err`, `uy_err` | `(318, 174)` | velocity uncertainties |
| `s` | `(318, 174)` | surface elevation |
| `bed` | `(318, 174)` | bed elevation |
| `bed_err`, `surf_err` | `(318, 174)` | geometry uncertainties |
| `xmin`, `xmax`, `ymin`, `ymax` | scalars | domain bounds (m) |

## Archive-ready extras

`totten_archive_vi_2022.npz` adds:

- aliases `thickness`, `surface`, `speed`
- explicit `x`, `y`, `X`, `Y` (500 m; `y` descending to match `load_snapshot`)
- metadata-only `cfg_json` (**no** `A` / `C` — sliding comes from VI configs)
- **no** `viscosity` (unknown on real Totten)

~47% of cells are finite geometry.

## Two basal-sliding cases

Same observation NPZ and **one shared PINN pretrain**; two VI physics end-members:

| Case | Config | `friction_C` | Checkdir |
|------|--------|--------------|----------|
| no sliding | [`Archive/configs/totten/run_torch_vi_only_totten_no_sliding.cfg`](../../../Archive/configs/totten/run_torch_vi_only_totten_no_sliding.cfg) | `100` | `checkpoints/torch_vi_only/totten/no_sliding/` |
| max sliding | [`Archive/configs/totten/run_torch_vi_only_totten_max_sliding.cfg`](../../../Archive/configs/totten/run_torch_vi_only_totten_max_sliding.cfg) | `0.001` | `checkpoints/torch_vi_only/totten/max_sliding/` |

```bash
cd Archive
python pretrain_solution_torch.py configs/totten/run_torch_pretrain_totten.cfg
python train_vi_only_torch.py configs/totten/run_torch_vi_only_totten_no_sliding.cfg
python train_vi_only_torch.py configs/totten/run_torch_vi_only_totten_max_sliding.cfg
```

Both VI configs use `eta_init=15`, `eta_prior_scale=0.08`, `kl_eta=0.15`, and `eta_eval_every=0` (no η scoring / no η early-stop).

QC maps: `outputs/figures/real/totten/`.
