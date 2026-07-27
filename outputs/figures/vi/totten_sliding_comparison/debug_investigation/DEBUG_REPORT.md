# Totten sequential VI — debugging investigation

## Executive summary

C cfg values preserved (True). Icepack SSA units consistent (True). Forward |τ_b| relative sensitivity to C: 16.4 (strong). Inferred η floored: frac(η≤1)≈0.59, max/min≈3.19. Primary failure mode: η posterior collapse to η_min — NOT a silent C drop and NOT a missing MPa·yr rescale of g. More epochs alone will not help.

## Task 1 — Does C propagate?

- `friction_C` read correctly: **True** (no=100.0, max=0.001)
- Passed into forward / basal drag / SSA residual / η gradients: **yes** (C is not learnable)

Pipeline:

- 1. ConfigParser optionxform=str preserves friction_C case (utilities_torch.ParameterClass)
- 2. pars.prior.friction_C set from cfg; lowercase alias friction_c also synced
- 3. Totten NPZ omits C — no cfg_json override of friction_C
- 4. icepack_ssa_constants(pars) → torch tensor friction_C
- 5. JointModel._physics_nll_ssa reads icepack['friction_C']
- 6. spinup_plastic_basal_drag(..., friction_C, ...) → basal_drag_{x,y}
- 7. SSA residual r = membrane_div(η) + τ_d − basal_drag(C)
- 8. physics_nll = −log N(r; σ); phys_scale * physics_nll enters total loss
- 9. Gradients: C is constant (no C grads); η VGP gets ∂L/∂η via membrane; C changes residual target
- 10. λ GP is NOT used in SSA basal drag (kl_lambda=0; plastic C law only)

## Task 2 — Sensitivity to C (forward, same weights)

- Checkpoint: `checkpoints/torch_vi_only/totten/no_sliding/model.pt` epoch 299
- Note: Fast forward probe on DSI: frozen PINN only; η held at exp(log(eta_init)+shift)≈1.83 MPa·yr.
- Basal drag mean |τ_b| change (C=100 vs 0.001): **16.4** relative
- SSA residual RMS change: **0.879** relative
- u_c mean ratio (C=100 / C=0.001): **1e-15**
- Δ approx momentum NLL: 0.416138

Per-C stats:
```json
{
  "C=100": {
    "friction_C_in_icepack": 100.0,
    "eta_center_used": 1.8297636996604427,
    "basal_drag_mean_abs": 0.025217000095854675,
    "basal_drag_mean_abs_grounded": 0.07506455842486973,
    "ssa_residual_rms": 0.06465900014893115,
    "approx_momentum_nll": 0.5806647639249277,
    "tau_c_mean": 0.039218857994360576,
    "frac_grounded_tau_c_pos": 0.3359375,
    "u_c_mean": 1.2837133751089163e-09,
    "driving_stress_rms": 0.02755794242779504,
    "membrane_div_rms": 0.019044652323406595,
    "speed_pred_mean": 867.9477052015911,
    "speed_pred_max": 2360.5734815003425,
    "eta_mean": 1.8297636996604427
  },
  "C=0.001": {
    "friction_C_in_icepack": 0.001,
    "eta_center_used": 1.8297636996604427,
    "basal_drag_mean_abs": 0.0014493377654170956,
    "basal_drag_mean_abs_grounded": 0.004314307766822982,
    "ssa_residual_rms": 0.034417915252794085,
    "approx_momentum_nll": 0.16452679032618275,
    "tau_c_mean": 0.039218857994360576,
    "frac_grounded_tau_c_pos": 0.3359375,
    "u_c_mean": 1283713.375108916,
    "driving_stress_rms": 0.02755794242779504,
    "membrane_div_rms": 0.019044652323406595,
    "speed_pred_mean": 867.9477052015911,
    "speed_pred_max": 2360.5734815003425,
    "eta_mean": 1.8297636996604427
  }
}
```

## Task 3 — Equation-loss units

Equation loss is dimensionally consistent in the icepack MPa system. Bare g=9.81 is NOT used in SSA; g and ρ are year-scaled so ρg = SI/1e6. No additional gravity rescaling is required after switching to exp η in MPa·yr.

- ρg (icepack) = 0.00899577 MPa/m (SI/1e6 = 0.00899577; consistent=True)
- Membrane check: H[m] * η[MPa·yr] * ∇u[1/yr] → MPa·m; ∇·(·) → MPa
- **No bare g=9.81 in SSA**; gravity is year-scaled. Do not rescale g further for MPa·yr η.

## Task 4 — Learning dynamics

- **no_sliding**: phys Δ(0→50)=-1.219e+04, phys Δ(150→end)=-3.869, ‖∇η‖ last/first=0.0001056, plateau=True, more epochs help? False
- **max_sliding**: phys Δ(0→50)=-1.218e+04, phys Δ(150→end)=-3.645, ‖∇η‖ last/first=9.098e-05, plateau=True, more epochs help? False

See `learning_dynamics.png`.

## Task 5 — Viscosity dynamic range

Posterior mass is heavily floored at η_min=1 MPa·yr (frac≈0.59). max/min≈3.19 (only 0.50 dex). In Pa·s the floor is ~3.16e13, so a ~1.5e13 Pa·s 'range' is a small absolute spread around the floor — not a multi-order physical viscosity field. This is posterior collapse / floor-pinning, not a healthy 1e10–1e15 Pa·s span.

```json
{
  "no_sliding": {
    "n": 26190,
    "min_MPa_yr": 1.0,
    "max_MPa_yr": 3.194359952010356,
    "mean_MPa_yr": 1.0993388535223745,
    "median_MPa_yr": 1.0,
    "std_MPa_yr": 0.18362912733440104,
    "max_over_min_ratio": 3.194359952010356,
    "frac_at_eta_min_1": 0.5859870179457808,
    "min_Pa_s": 31557600000000.0,
    "max_Pa_s": 100806333621562.0,
    "mean_Pa_s": 34692495803917.684,
    "max_over_min_ratio_Pa_s": 3.194359952010356,
    "expected_glacier_range_Pa_s": [
      10000000000.0,
      1000000000000000.0
    ],
    "span_orders_of_magnitude_log10": 0.5043838524298512
  },
  "max_sliding": {
    "n": 26190,
    "min_MPa_yr": 1.0,
    "max_MPa_yr": 2.2268275346585384,
    "mean_MPa_yr": 1.0855427767073282,
    "median_MPa_yr": 1.0,
    "std_MPa_yr": 0.1508978795101036,
    "max_over_min_ratio": 2.2268275346585384,
    "frac_at_eta_min_1": 0.5990454371897671,
    "min_Pa_s": 31557600000000.0,
    "max_Pa_s": 70273332607740.29,
    "mean_Pa_s": 34257124730219.18,
    "max_over_min_ratio_Pa_s": 2.2268275346585384,
    "expected_glacier_range_Pa_s": [
      10000000000.0,
      1000000000000000.0
    ],
    "span_orders_of_magnitude_log10": 0.3476865827069023
  }
}
```

## Task 6 — Why no≈max sliding?

- Nearly identical optimization outcomes: **True**

| metric | no (C=100) | max (C=0.001) | Δ |
|---|---:|---:|---:|
| initial_loss | 12471.4 | 12456.2 | 15.1344 |
| final_loss | -6.88369 | -9.54153 | 2.65784 |
| final_phys | -9.07918 | -11.7284 | 2.64926 |
| final_data | 1.91085 | 1.91085 | 0 |
| final_kl | 0.284645 | 0.276061 | 0.00858323 |
| grad_vgp_eta_last | 0.756068 | 0.651754 | 0.104314 |
| inferred_mean_eta | 1.09934 | 1.08554 | 0.0137961 |
| inferred_eta_std | 0.183629 | 0.150898 | 0.0327312 |
| inferred_eta_max_min_ratio | 3.19436 | 2.22683 | 0.967532 |
| frac_at_floor | 0.585987 | 0.599045 | -0.0130584 |

## Most likely root cause

C is correctly wired and DOES change SSA basal drag / residuals (forward |τ_b| relative Δ≈16.4 between C=100 and 0.001). The nearly identical no/max sliding *results* are instead explained by η posterior collapse onto η_min=1 (median=1, ~60% at floor; eta_log_shift≈−2.1). With η clipped, membrane stress cannot express C-dependent spatial structure, so inferred maps and late-training losses converge to the same floored solution. Units are consistent; this is not a missing-C bug or a bare-g=9.81 bug.

## Recommended code changes (only after cause ID)

1. Do not "fix" units by changing g=9.81 in SSA — icepack scaling is already correct.
2. Instrument training to log mean |basal_drag|, mean τ_c, mean u_c, friction_C, and frac(η≤η_min) each epoch.
3. Address η floor collapse before trusting sliding end-members: re-center prior (eta_init / eta_log_shift / eta_prior) and/or raise η_min carefully so the field is not clipped.
4. Only ~34% of probe points have τ_c>0. Prefer grounded-masked diagnostics, but C sensitivity is already strong — prioritize η-floor fixes.
5. More epochs alone will not help: physics loss and ‖∇η‖ plateau by ~epoch 100–150.
6. Optional ablation: ssa_use_inferred_eta=False to confirm basal-C signal in physics NLL without η absorbing everything into the floor.
