# Totten viscosity posterior collapse — prior diagnosis (Task 1)

## Parameterization (unchanged SSA)

```
η = materialize( log(η_init) + η_log_shift + θ )
```

Default bound mode `log_clamp`:

```
η_log = clamp(z, log(η_min), log(η_max))
η = exp(η_log)
```

Optional `eta_bound_mode = softplus_floor` (behind config flag):

```
η = η_min + softplus(exp(z) − η_min)   # then soft-cap at η_max
```

## Prior terms

| Knob | Location | Totten baseline | Role |
|------|----------|-----------------|------|
| `eta_init` | `[prior]` | 15 | Log-center of the field before shift |
| `eta_log_shift` | learned `nn.Parameter` | → ≈ −2.1 | Global log offset (optimizer-driven) |
| `eta_prior_mean` | `[train]` | `'auto'` (= η_init) | Soft Gaussian target in log space |
| `eta_prior_std` | `[train]` | 1.5 | Soft-prior width (log η) |
| `eta_prior_scale` | `[train]` | 0.08 | Soft-prior weight |
| `kl_eta` | `[prior]` | 0.15 | Sparse-GP KL weight |
| `eta_min` / `eta_max` | `[prior]` | 1 / 1e6 | Bounds |

Soft prior pulls

`log(η_init) + η_log_shift + E[θ]  →  log(eta_prior_mean)`.

With `auto`, that is equivalent to pulling `η_log_shift + E[θ] → 0`.

## Where clipping happens

Hard clamp is **inside** the autograd graph (`Tensor.clamp` on `η_log` before `exp`).
At the floor, `∂η/∂z = 0` for the clamped coordinate → the optimizer cannot push floored pixels back up through the physics residual path.

## Does the optimizer want η≈1, or does clipping force it?

Both:

1. **Physics preference:** with a frozen PINN, SSA residuals are reduced by lowering membrane viscosity; `η_log_shift` dives negative (≈ −2.1).
2. **Clip enforcement:** once `log(η_init)+shift+θ < log(η_min)`, values stick at `η_min=1` with zero local gradient (~59% of Totten).

Effective center after training: `η_init * exp(shift) ≈ 15 * exp(−2.1) ≈ 1.83`, so many spatial samples sit below the floor.

## Conclusion

Do **not** change SSA / g / C. Raise or re-center the prior (`eta_init` sweep) and/or use `softplus_floor` so the bound does not kill gradients. Instrument `frac(η≤η_min)` every few epochs.
