# icepack SSA equations

Reference for the **Shallow Shelf / Shallow Stream Approximation (SSA)** as implemented in [icepack](https://github.com/icepack/icepack). This document matches the v1.0 formulation in Shapero et al. (2021, GMD) and the installed source modules:

- `icepack/models/ice_stream.py` — grounded fast flow (`IceStream`)
- `icepack/models/ice_shelf.py` — floating ice shelves (`IceShelf`)
- `icepack/models/viscosity.py` — Glen law and membrane stress
- `icepack/models/friction.py` — Weertman basal and sidewall friction
- `icepack/constants.py` — units and default exponents

The Ice Dynamics spin-up notebooks (`notebooks/spinup/spinupNewFull-*.ipynb`) use **`IceStream`** via `icepack.solvers.FlowSolver`.

> **Math rendering:** Display equations use `$$...$$` and inline symbols use `$...$` (GitHub, VS Code, and most Jupyter markdown renderers).

---

## Overview

icepack solves SSA through a **variational (action) principle**, not by assembling a PDE residual directly. The horizontal ice velocity $\mathbf{u}$ is found by minimizing an action functional $J$ (units: power = energy/time). The Euler–Lagrange equations of $J$ are equivalent to the standard SSA momentum balance:

$$
\frac{\delta J}{\delta \mathbf{u}} = 0.
$$

| icepack class | Name in literature | Regime |
|---------------|-------------------|--------|
| `IceStream` | Shallow **Stream** Approximation (SSA, grounded) | Fast grounded ice with basal sliding |
| `IceShelf` | Shallow **Shelf** Approximation (SSA, floating) | Ice shelves in hydrostatic balance |

Both are 2D, depth-averaged models derived from the Blatter–Pattyn (first-order) system by assuming **plug flow** (horizontal extension dominates over vertical shear):

$$
\left|\frac{\partial \mathbf{u}}{\partial z}\right| \ll \left|\frac{\partial \mathbf{u}}{\partial x}\right|.
$$

See MacAyeal (1989) and Shapero et al. (2021, §2.2.2).

---

## Units and constants

icepack uses **meters, years, megapascals (MPa)** (same convention as Elmer/Ice).

| Symbol | Meaning | Default in icepack |
|--------|---------|-------------------|
| $n$ | Glen flow-law exponent | $3$ (`glen_flow_law`) |
| $m$ | Weertman sliding exponent | $3$ (`weertman_sliding_law`) |
| $\dot\varepsilon_{\min}$ | Strain-rate regularization | $10^{-5}\,\mathrm{yr}^{-1}$ (`strain_rate_min`) |
| $\rho_I$ | Ice density | $917\,\mathrm{kg\,m^{-3}}$ (scaled in code) |
| $\rho_W$ | Seawater density | $1024\,\mathrm{kg\,m^{-3}}$ (scaled in code) |
| $g$ | Gravity | $9.81\,\mathrm{m\,yr^{-2}}$ (scaled in code) |
| $A$ | Fluidity (Glen rate factor) | User field (`fluidity` argument) |
| $C$ | Basal friction coefficient | User field (`friction` argument) |

---

## Field variables

| Symbol | Name | Role |
|--------|------|------|
| $\mathbf{u}$ | Horizontal velocity (2D vector) | Unknown in diagnostic solve |
| $h$ | Ice thickness | Input geometry |
| $s$ | Surface elevation | Input geometry |
| $b$ | Bed elevation | Grounded: $s = b + h$ |
| $A$ | Fluidity | Rheology parameter (inverse viscosity scale) |
| $C$ | Basal friction coefficient | Sliding law strength (grounded only) |

In this project, spin-up saves **effective viscosity** $\eta$ to the grid NPZ as a **diagnostic** of the solved velocity and fluidity — it is not an independent input to the SSA solve.

---

## Strain rate and Glen constitutive law

### Strain-rate tensor

In plan-view ($xy$) models, icepack defines the horizontal strain-rate tensor as the symmetric gradient (`icepack.calculus.sym_grad`):

$$
\dot{\boldsymbol{\varepsilon}} = \frac{1}{2}\left(\nabla \mathbf{u} + \nabla \mathbf{u}^{\mathsf{T}}\right).
$$

### Effective strain rate (regularized)

To avoid singularities at zero strain rate:

$$
\dot\varepsilon_e = \sqrt{\frac{1}{2}\left(\dot{\boldsymbol{\varepsilon}} : \dot{\boldsymbol{\varepsilon}} + (\mathrm{tr}\,\dot{\boldsymbol{\varepsilon}})^2 + \dot\varepsilon_{\min}^2\right)}.
$$

### Glen flow law

$$
\dot{\boldsymbol{\varepsilon}} = A\,|\boldsymbol{\tau}|^{n-1}\,\boldsymbol{\tau},
$$

where $\boldsymbol{\tau}$ is the deviatoric stress tensor and $A$ is the temperature-dependent fluidity (scalar field in icepack).

### Membrane stress tensor

For depth-averaged SSA, icepack uses a **membrane stress** tensor $\mathbf{M}$ (`icepack.models.viscosity.membrane_stress`):

$$
\mathbf{M} = 2\mu\left(\dot{\boldsymbol{\varepsilon}} + (\mathrm{tr}\,\dot{\boldsymbol{\varepsilon}})\,\mathbf{I}\right),
\qquad
\mu = \frac{1}{2}\,A^{-1/n}\,\dot\varepsilon_e^{\,1/n - 1}.
$$

The depth-averaged Cauchy stress is related to $\mathbf{M}$ through the vertical integration that produces the SSA system.

---

## Action functional (general form)

All icepack diagnostic models minimize:

$$
J = J_{\mathrm{visc}} + J_{\mathrm{fric}} + J_{\mathrm{side}} - J_{\mathrm{grav}} - J_{\mathrm{term}} + J_{\mathrm{penalty}}.
$$

(terms absent in a given model are omitted)

| Term | Physical meaning |
|------|------------------|
| $J_{\mathrm{visc}}$ | Viscous dissipation (membrane / Glen) |
| $J_{\mathrm{fric}}$ | Basal sliding friction (grounded only) |
| $J_{\mathrm{side}}$ | Sidewall drag along fjord walls |
| $J_{\mathrm{grav}}$ | Gravitational driving (surface slope or buoyancy) |
| $J_{\mathrm{term}}$ | Stress work at calving / grounding line |
| $J_{\mathrm{penalty}}$ | Penalty for normal flow at sidewalls |

---

## Viscous term (shared by `IceStream` and `IceShelf`)

From `viscosity_depth_averaged()`:

$$
J_{\mathrm{visc}} = \int_\Omega \frac{2n}{n+1}\, h\, A^{-1/n}\, \dot\varepsilon_e^{\,1/n + 1}\,\mathrm{d}x.
$$

The GMD paper (Eq. 10) writes an equivalent form:

$$
J_{\mathrm{visc}} = \int_\Omega \frac{n}{n+1}\, h\, A^{-1/n}\, |\dot{\boldsymbol{\varepsilon}}|^{1/n + 1}\,\mathrm{d}x.
$$

The two expressions use slightly different tensor-norm conventions but represent the same Glen dissipation.

With $n = 3$:

$$
J_{\mathrm{visc}} = \int_\Omega \frac{3}{2}\, h\, A^{-1/3}\, \dot\varepsilon_e^{\,4/3}\,\mathrm{d}x.
$$

---

## Grounded SSA — `IceStream`

### Total action

$$
J = J_{\mathrm{visc}} + J_{\mathrm{fric}} + J_{\mathrm{side}} - J_{\mathrm{grav}} - J_{\mathrm{term}} + J_{\mathrm{penalty}}.
$$

### Basal friction (Weertman law)

Basal shear stress:

$$
\boldsymbol{\tau}_b = -C\,|\mathbf{u}|^{1/m - 1}\,\mathbf{u}.
$$

Friction contribution to the action (`bed_friction`):

$$
\begin{aligned}
J_{\mathrm{fric}}
&= -\frac{m}{m+1} \int_\Omega \boldsymbol{\tau}_b \cdot \mathbf{u}\,\mathrm{d}x \\
&= \frac{m}{m+1} \int_\Omega C\,|\mathbf{u}|^{1/m + 1}\,\mathrm{d}x.
\end{aligned}
$$

With $m = 3$, basal drag magnitude scales as $C\,|\mathbf{u}|^2$.

Smaller $C$ → more sliding. The Ice Dynamics spin-up cases use small $C$ (more sliding) vs large $C$ (effectively no sliding).

### Gravitational driving

$$
J_{\mathrm{grav}} = -\int_\Omega \rho_I\, g\, h\, \nabla s \cdot \mathbf{u}\,\mathrm{d}x.
$$

This is the work done by the driving stress $\rho_I g h \nabla s$ against the ice flow.

### Calving / marine terminus (grounded)

$$
J_{\mathrm{term}} = \int_\Gamma \left(\frac{1}{2}\rho_I g h^2 - \frac{1}{2}\rho_W g d^2\right) \mathbf{u} \cdot \boldsymbol{\nu}\,\mathrm{d}\gamma,
$$

where:

- $\Gamma$ is the ice front boundary
- $\boldsymbol{\nu}$ is the outward unit normal
- $d = \min(s - h,\, 0)$ is water depth (sea level at $z = 0$)

### Sidewall friction (optional)

On boundary IDs marked as sidewalls:

$$
J_{\mathrm{side}} = -\frac{m}{m+1} \int_\Gamma h\,\boldsymbol{\tau}_s(\mathbf{u}_t) \cdot \mathbf{u}_t\,\mathrm{d}\gamma,
$$

where $\mathbf{u}_t$ is the velocity tangent to the wall and $\boldsymbol{\tau}_s$ has the same Weertman form as basal friction with coefficient $C_s$.

### Strong-form momentum balance (grounded)

Taking the first variation of $J$ with respect to $\mathbf{u}$ gives the standard MacAyeal SSA system:

$$
\nabla \cdot (h\boldsymbol{\sigma}) + \rho_I g h\,\nabla s - \boldsymbol{\tau}_b = \mathbf{0},
$$

where $\boldsymbol{\sigma}$ is the depth-averaged membrane stress derived from Glen's law, and

$$
\boldsymbol{\tau}_b = C\,|\mathbf{u}|^{1/m-1}\,\mathbf{u}.
$$

In words: **divergence of membrane stress** + **driving stress** = **basal drag**.

---

## Floating SSA — `IceShelf`

For floating ice:

- Basal friction $C = 0$
- Hydrostatic surface: $s = \left(1 - \rho_I/\rho_W\right) h$
- icepack uses an integrated-by-parts form where the shelf gravity and terminus terms are related

Define the **buoyancy-adjusted density**:

$$
\varrho = \rho_I\left(1 - \frac{\rho_I}{\rho_W}\right).
$$

### Buoyancy / gravity

$$
J_{\mathrm{grav}} = -\frac{1}{2}\int_\Omega \varrho\, g\, \nabla(h^2) \cdot \mathbf{u}\,\mathrm{d}x.
$$

### Terminus

$$
J_{\mathrm{term}} = \frac{1}{2}\int_\Gamma \varrho\, g\, h^2\, \mathbf{u} \cdot \boldsymbol{\nu}\,\mathrm{d}\gamma.
$$

### Strong-form momentum balance (floating)

$$
\nabla \cdot (h\boldsymbol{\sigma}) + \varrho\, g\, h\,\nabla h = \mathbf{0}.
$$

This is the shelf SSA balance: membrane stress divergence balances the buoyancy-driven driving stress.

---

## Prognostic thickness equation

SSA diagnostic solves are coupled to thickness evolution via the continuity equation (`icepack.models.transport.Continuity`):

$$
\frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) = \dot a_s - \dot a_b.
$$

| Symbol | Meaning |
|--------|---------|
| $\dot a_s$ | Surface mass balance |
| $\dot a_b$ | Basal mass balance |

icepack's prognostic solver truncates $h$ to zero where it becomes negative (approximate ice margin tracking).

---

## Boundary conditions

| Type | Implementation |
|------|----------------|
| **Inflow Dirichlet** | Prescribe $\mathbf{u}$ where ice flows in |
| **Calving / terminus** | Natural (Neumann) via $J_{\mathrm{term}}$ |
| **Sidewalls** | No normal flow via penalty; tangential drag via $J_{\mathrm{side}}$ |
| **Outflow** | Natural where ice flows out |

---

## Numerical solution in icepack

```python
import icepack

model = icepack.models.IceStream()          # or IceShelf()
solver = icepack.solvers.FlowSolver(model)

u = solver.diagnostic_solve(
    velocity=u0,           # initial guess
    thickness=h,
    surface=s,
    fluidity=A,            # Glen rate factor
    friction=C,            # Weertman coefficient (IceStream only)
    ice_front_ids=(...),   # optional boundary markers
    side_wall_ids=(...),
)
```

`FlowSolver` minimizes $J$ using Newton's method on the weak form. The action is convex near steady state, which gives robust convergence.

Thickness is updated separately with `prognostic_solve` / transport stepping.

---

## Mapping to this project

| icepack quantity | Spin-up NPZ field | Notes |
|------------------|-------------------|-------|
| $\mathbf{u}$ | `ux`, `uy`, `speed`, `velocity` | Diagnostic SSA solution |
| $h$ | `thickness` | |
| $s$ | `surface` | |
| $A$ | `A`, `A_inv` | Fluidity |
| $C$ | `cfg_json["C"]` | Sliding coefficient |
| $\eta$ (effective viscosity) | `viscosity` | Post-processed diagnostic, not SSA input |

The VI workflow treats $\eta$ (or $\log\eta$) as an inference target; the forward model in spin-up infers it from $\mathbf{u}$ and $A$ via Glen's law after the SSA solve.

---

## SSA vs other icepack models

| Model | Class | Dominant physics | Use |
|-------|-------|------------------|-----|
| SIA | `ShallowIce` | Vertical shear | Ice sheet interior |
| SSA (grounded) | `IceStream` | Membrane + sliding | Fast outlets, MISMIP+ |
| SSA (floating) | `IceShelf` | Membrane + buoyancy | Ice shelves |
| First-order | `HybridModel` | Shear + plug modes | Higher fidelity 3D |

---

## icepack2 (dual SSA)

The separate [icepack/icepack2](https://github.com/icepack/icepack2) repository implements a **dual (mixed) SSA** formulation with membrane and basal stresses as explicit unknowns. That formulation remains solvable when thickness goes to zero. The main `icepack` v1 classes (`IceStream`, `IceShelf`) use the **primal velocity formulation** described above.

---

## References

1. **Shapero, D., Badge, J., & Hoffman, M.** (2021). icepack: a new glacier flow modeling package in Python, version 1.0. *Geoscientific Model Development*, 14, 4593–4616. [https://gmd.copernicus.org/articles/14/4593/2021/](https://gmd.copernicus.org/articles/14/4593/2021/)
2. **MacAyeal, D. R.** (1989). Large-scale ice flow over a viscous basal sediment: theory and application to Ice Stream B, Antarctica. *Journal of Geophysical Research*, 94(B4), 4071–4087.
3. **Greve, R., & Blatter, H.** (2009). *Dynamics of Ice Sheets and Glaciers*. Springer (analytical shelf/stream solutions).
4. **Cuffey, K. M., & Paterson, W. S. B.** (2010). *The Physics of Glaciers*. Elsevier (Glen flow law).
5. **icepack source**: [https://github.com/icepack/icepack](https://github.com/icepack/icepack)

---

## Quick equation summary

**Glen law** ($n=3$):

$$
\dot{\boldsymbol{\varepsilon}} = A\,|\boldsymbol{\tau}|^{n-1}\,\boldsymbol{\tau}.
$$

**Grounded SSA (strong form)** ($m=3$):

$$
\nabla \cdot (h\boldsymbol{\sigma}) + \rho_I g h \nabla s = C\,|\mathbf{u}|^{1/m-1}\,\mathbf{u}.
$$

**Floating SSA (strong form)**:

$$
\nabla \cdot (h\boldsymbol{\sigma}) + \varrho g h \nabla h = \mathbf{0},
\qquad
\varrho = \rho_I\left(1 - \frac{\rho_I}{\rho_W}\right).
$$

**Thickness evolution**:

$$
\frac{\partial h}{\partial t} + \nabla \cdot (h\mathbf{u}) = \dot a_s - \dot a_b.
$$

**icepack viscous action**:

$$
J_{\mathrm{visc}} = \int_\Omega \frac{2n}{n+1}\, h\, A^{-1/n}\, \dot\varepsilon_e^{\,1/n+1}\,\mathrm{d}x.
$$
