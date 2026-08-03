# Totten C-sensitivity comparison report

Ranked by grounded η distinguishability: `score = 10*(0.97 − corr_grounded) + mean|Δη|_grounded`.

SSA / Icepack equations were **not** modified; only cfg flags (physics region weights, η shift freeze, GP flexibility).

| Rank | Experiment | corr (full) | corr (grounded) | corr (floating) | mean\|Δη\| gnd | max\|Δη\| gnd | phys no / max | ELBO no / max |
|-----:|------------|------------:|----------------:|----------------:|---------------:|--------------:|--------------:|--------------:|
| 1 | `unfreeze_float0` | 0.4853 | 0.4429 | 0.5626 | 2.569 | 4.791 | None / None | None / None |
| 2 | `phys_w10_Cextreme` | 0.9425 | 0.9015 | 0.9734 | 0.1107 | 2.089 | None / None | None / None |
| 3 | `phys_w10` | 0.9493 | 0.9123 | 0.9777 | 0.1047 | 1.997 | None / None | None / None |
| 4 | `phys_w5` | 0.9523 | 0.9147 | 0.9803 | 0.1014 | 2.072 | None / None | None / None |
| 5 | `phys_w2` | 0.9589 | 0.9217 | 0.9854 | 0.09373 | 2.116 | None / None | None / None |
| 6 | `baseline` | 0.9659 | 0.9299 | 0.9903 | 0.08492 | 2.02 | None / None | None / None |
| 7 | `gp_short_ls` | 0.9719 | 0.9409 | 0.9922 | 0.07856 | 1.357 | None / None | None / None |
| 8 | `freeze_shift` | 0.9963 | 0.9907 | 0.9994 | 0.1883 | 4.839 | None / None | None / None |
| 9 | `gp_flex` | 0.9973 | 0.9934 | 0.9996 | 0.021 | 0.239 | None / None | None / None |

## Interpretation guide

- If **corr_floating ≫ corr_grounded**, global metrics were masking C effects.
- If **freeze_shift** drops correlation, the global intercept was absorbing residuals.
- If **phys_w\*** helps, floating-dominated averaging was diluting basal-friction signal.
- If **gp_flex / gp_short_ls** helps, the prior was too smooth to express C structure.

Figures: `outputs/figures/vi/totten_c_sensitivity/ranking_overview.png`
Table: `outputs/figures/vi/totten_c_sensitivity/comparison_table.csv`

