import os
import json
import time as walltime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import icepack
import icepack.plot
import firedrake
from firedrake import (
    exp,
    sin,
    max_value,
    Constant,
    sqrt,
    inner,
    dx,
    dS,
    as_vector,
    Function,
    PointNotInDomainError,
)
from firedrake.exceptions import ConvergenceError
from icepack.models.viscosity import viscosity_depth_averaged
from firedrake import sym, grad, tr, inner, sqrt, Constant
from icepack.constants import glen_flow_law as n

from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    gravity as g,
    weertman_sliding_law as m,
)

from meshpy import triangle
from mpi4py import MPI
import tqdm


# ------------------------------------------------------------
# config
# ------------------------------------------------------------
# Set TEST_MODE = True for a shorter smoke test (200 yr/stage, ramp scaled).
# Set TEST_MODE = False for the production 10500 yr spin-up.
TEST_MODE = False
from pathlib import Path


def project_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "outputs" / "spinup").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = project_root()

cfg = {
    # Domain / MISMIP+ geometry
    "Lx": 640e3,
    "Ly": 80e3,
    "Ny": 15,

    # Physical parameters used by the spinup solve
    "A": 80, # pre-factor (softer ice → lower η vs A=20 baseline)
    # End-member more sliding (small C).
    "C": 1e-3,
    "C_start": 1e-2,  # moderate sliding; ramped during spin-up for stability
    "C_ramp_time": 4000.0,
    "a": 0.3,  # accumulation

    # Coarse-grid spin-up (4000 yr ramp + 6500 yr at target C per stage)
    "coarse_total_time": 10500.0,
    "coarse_dt": 0.25,

    # Fine-grid continuation
    "fine_total_time": 10500.0,
    "fine_dt": 0.25,
    "primary_solver": "petsc",  # petsc <-> icepack fallback each step

    # Solver/debug verbosity
    "monitor_snes": False,
    "step_print_every": 10,

    # Refinement 
    "shrink": 8,
    "exponent": 2,

    # Output 
    "case_id": "more_sliding_A80",
    "outdir": "",
    "save_dir": "",
    "output_stem": "SteadyState_more_sliding_A80_10500yr_ramp4000_1refine",
    "grid_resolution": 500.0, # m

    # A-field. the solver uses constant A.
    "Ax": 0.0,
    "Ay": 0.0,
    "A_field_saved_value": 80.0,
}

TEST_STAGE_YEARS = 200.0
PRODUCTION_STAGE_YEARS = cfg["coarse_total_time"]
PRODUCTION_C_RAMP_TIME = cfg["C_ramp_time"]

TEST_OVERRIDES = {
    "coarse_total_time": TEST_STAGE_YEARS,
    "fine_total_time": TEST_STAGE_YEARS,
    "coarse_dt": 0.25,
    "fine_dt": 0.25,
    "C_ramp_time": TEST_STAGE_YEARS * PRODUCTION_C_RAMP_TIME / PRODUCTION_STAGE_YEARS,
    "step_print_every": 10,
}

MODE = "test" if TEST_MODE else "production"
cfg["save_dir"] = str(PROJECT_ROOT / "outputs" / "spinup" / MODE / cfg["case_id"])
cfg["outdir"] = str(PROJECT_ROOT / "outputs" / "figures" / MODE / cfg["case_id"])

if TEST_MODE:
    cfg.update(TEST_OVERRIDES)
    cfg["test_mode"] = True
    cfg["output_stem"] = (
        cfg["output_stem"].replace(
            f"_{int(PRODUCTION_STAGE_YEARS)}yr_",
            f"_test_{int(TEST_STAGE_YEARS)}yr_",
        )
        if f"_{int(PRODUCTION_STAGE_YEARS)}yr_" in cfg["output_stem"]
        else f"{cfg['output_stem']}_test"
    )
else:
    cfg["test_mode"] = False

if MPI.COMM_WORLD.rank == 0:
    mode_label = "TEST" if TEST_MODE else "PRODUCTION"
    print(
        f"\n=== Spin-up mode: {mode_label} ===\n"
        f"stage time = {cfg['coarse_total_time']} yr, dt = {cfg['coarse_dt']} yr, "
        f"C ramp = {cfg.get('C_start', cfg['C'])} -> {cfg['C']} over "
        f"{cfg.get('C_ramp_time', 0.0)} yr\n"
        f"outputs -> {cfg['outdir']}/, stem = {cfg['output_stem']}\n",
        flush=True,
    )

Lx, Ly = cfg["Lx"], cfg["Ly"]
DOMAIN_AREA = Lx * Ly

OUTDIR = cfg["outdir"]
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(cfg["save_dir"], exist_ok=True)

COARSE_TOTAL_TIME = cfg["coarse_total_time"]
COARSE_DT = cfg["coarse_dt"]

FINE_TOTAL_TIME = cfg["fine_total_time"]
FINE_DT = cfg["fine_dt"]

MONITOR_SNES = cfg["monitor_snes"]
STEP_PRINT_EVERY = cfg["step_print_every"]

SPINUP_HISTORY: list[dict] = []

SHRINK = cfg["shrink"]
EXPONENT = cfg["exponent"]

C_TARGET = float(cfg["C"])
C_START = float(cfg.get("C_start", cfg["C"]))
C_RAMP_TIME = float(cfg.get("C_ramp_time", 0.0))
SIMULATION_ELAPSED_YEARS = 0.0  # cumulative time for C ramp across stages
PRIMARY_SOLVER = cfg.get("primary_solver", "petsc")


# ------------------------------------------------------------
# Physics and geometry
# ------------------------------------------------------------

def mismip_bed_topography(mesh):
    x, y = firedrake.SpatialCoordinate(mesh)

    x_c = Constant(300e3)
    X = x / x_c

    B_0 = Constant(-150)
    B_2 = Constant(-728.8)
    B_4 = Constant(343.91)
    B_6 = Constant(-50.57)
    B_x = B_0 + B_2 * X**2 + B_4 * X**4 + B_6 * X**6

    f_c = Constant(4e3)
    d_c = Constant(500)
    w_c = Constant(24e3)

    B_y = d_c * (
        1 / (1 + exp(-2 * (y - Ly / 2 - w_c) / f_c)) +
        1 / (1 + exp(+2 * (y - Ly / 2 + w_c) / f_c))
    )

    z_deep = Constant(-720)
    return max_value(B_x + B_y, z_deep)




def get_A_field(mesh, Lx, Ly, Ax, Ay):
    """A-field."""
    x, y = firedrake.SpatialCoordinate(mesh)
    A_y = Ay * sin(y / Ly * np.pi) + 9
    A_x = Ax * sin(x / Lx * np.pi) + 1
    return A_x + A_y


def viscosity(**kwargs):
    """Depth-averaged effective viscosity for gridded diagnostic output."""
    return viscosity_depth_averaged(
        velocity=kwargs["velocity"],
        thickness=kwargs["thickness"],
        fluidity=kwargs["fluidity"],
    )

def effective_viscosity(velocity, fluidity):
    eps = sym(grad(velocity))
    eps_eff = sqrt(
        0.5 * (inner(eps, eps) + tr(eps)**2) + Constant(1e-30)
    )
    return 0.5 * fluidity**(-1 / n) * eps_eff**(1 / n - 1)

def friction(**kwargs):
    variables = ("velocity", "thickness", "surface", "friction")
    u, h, s, C = map(kwargs.get, variables)

    p_W = ρ_W * g * max_value(0, -(s - h))
    p_I = ρ_I * g * h
    N = max_value(0, p_I - p_W)
    τ_c = N / 2

    u_c = (τ_c / C) ** m
    u_b = sqrt(inner(u, u))

    phi = τ_c * (
        (u_c**(1 / m + 1) + u_b**(1 / m + 1))**(m / (m + 1)) - u_c
    )

    sliding_scale = Constant(1e0)   

    return sliding_scale * phi


A = Constant(cfg["A"])
C = Constant(C_START)
a = Constant(cfg["a"])

model = icepack.models.IceStream(friction=friction)


def ramped_friction_value(elapsed_years):
    """Log-linear ramp from C_START to C_TARGET."""
    if C_RAMP_TIME <= 0.0 or elapsed_years >= C_RAMP_TIME:
        return C_TARGET
    if C_START <= 0.0 or C_TARGET <= 0.0:
        raise ValueError("C_START and C_TARGET must be positive for log ramping.")
    alpha = float(elapsed_years) / float(C_RAMP_TIME)
    return float(np.exp((1.0 - alpha) * np.log(C_START) + alpha * np.log(C_TARGET)))


def assign_ramped_C(elapsed_years):
    C.assign(ramped_friction_value(elapsed_years))


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def subplots():
    fig, axes = icepack.plot.subplots()
    axes.set_aspect(2)
    axes.set_xlim((0, Lx))
    axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors):
    return fig.colorbar(colors, fraction=0.012, pad=0.025)


def save_mesh_plot(mesh, filename, title=None):
    fig, axes = subplots()
    firedrake.triplot(mesh, axes=axes)
    if title is not None:
        axes.set_title(title)
    try:
        axes.legend(loc="upper right")
    except Exception:
        pass
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_field_plot(field, filename, title=None):
    fig, axes = subplots()
    colors = firedrake.tripcolor(field, axes=axes)
    colorbar(fig, colors)
    if title is not None:
        axes.set_title(title)
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_refined_mesh_plot(fine_mesh, coarse_high_fields, Q_high, filename):
    fig, axes = icepack.plot.subplots()
    axes.set_xlim((350e3, 550e3))
    axes.set_ylim((0, Ly))
    axes.get_yaxis().set_visible(False)

    s = coarse_high_fields["surface"]
    h = coarse_high_fields["thickness"]

    height_above_flotation = Function(Q_high).interpolate(
        s - (1 - ρ_I / ρ_W) * h
    )

    try:
        levels = [0, 1, 10]
        firedrake.tricontour(height_above_flotation, levels=levels, axes=axes)
    except Exception as err:
        print(
            f"Warning: grounding-line contour plot failed: {repr(err)}",
            flush=True,
        )

    firedrake.triplot(fine_mesh, axes=axes)
    axes.set_title("Refined mesh near grounding line")
    fig.savefig(os.path.join(OUTDIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Solver helpers
# ------------------------------------------------------------

def make_solver(kind, monitor=False):
    """
    kind = "icepack" or "petsc"

    The boundary ids follow MeshPy rectangle:
      1: bottom wall
      2: right / outflow
      3: top wall
      4: left / inflow
    """
    base_opts = {
        "dirichlet_ids": [4],
        "side_wall_ids": [1, 3],
    }

    if kind == "icepack":
        opts = {
            **base_opts,
            "diagnostic_solver_type": "icepack",
            "diagnostic_solver_parameters": {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "tolerance": 1e-5,
                "max_iterations": 1000,
            },
        }

    elif kind == "petsc":
        diagnostic_solver_parameters = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "cp",
            "snes_max_it": 1000,
            "snes_rtol": 1e-5,
            "snes_atol": 1e-7,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        if monitor:
            diagnostic_solver_parameters.update(
                {
                    "snes_monitor": None,
                    "snes_converged_reason": None,
                    "ksp_converged_reason": None,
                }
            )

        opts = {
            **base_opts,
            "diagnostic_solver_type": "petsc",
            "diagnostic_solver_parameters": diagnostic_solver_parameters,
        }

    else:
        raise ValueError(f"Unknown solver kind: {kind}")

    return icepack.solvers.FlowSolver(model, **opts)


def diagnostic_solve_with_fallback(
    primary_solver,
    fallback_solver,
    *,
    velocity,
    thickness,
    surface,
    stage_name,
):
    velocity_before = velocity.copy(deepcopy=True)
    last_error = None

    for label, solver in (
        ("primary", primary_solver),
        ("fallback", fallback_solver),
    ):
        if solver is None:
            continue

        guess = velocity if label == "primary" else velocity_before
        try:
            return solver.diagnostic_solve(
                velocity=guess,
                thickness=thickness,
                surface=surface,
                fluidity=A,
                friction=C,
            )
        except ConvergenceError as err:
            last_error = err
            print(
                f"\n{stage_name}: {label} diagnostic solve failed; "
                f"trying next solver. Error: {repr(err)}",
                flush=True,
            )

    raise last_error


def log_spaced_friction_values(c_target, *, c_start=None, n_steps=6):
    """Log-uniform C values from c_start to c_target (inclusive)."""
    c_start = float(C_START if c_start is None else c_start)
    c_target = float(c_target)
    if abs(np.log(c_target) - np.log(c_start)) < 1e-12:
        return [c_target]
    alphas = np.linspace(0.0, 1.0, int(n_steps))
    return [
        float(np.exp((1.0 - alpha) * np.log(c_start) + alpha * np.log(c_target)))
        for alpha in alphas
    ]


def diagnostic_solve_robust(
    primary_solver,
    fallback_solver,
    *,
    velocity,
    thickness,
    surface,
    stage_name,
    friction_continuation=False,
):
    """Diagnostic solve with optional log-spaced friction continuation."""
    target_c = float(C)

    if not friction_continuation:
        return diagnostic_solve_with_fallback(
            primary_solver,
            fallback_solver,
            velocity=velocity,
            thickness=thickness,
            surface=surface,
            stage_name=stage_name,
        )

    # Projected CG1 fields can be a poor Newton guess at target C on CG2.
    c_values = log_spaced_friction_values(target_c, n_steps=10)
    u = velocity
    for idx, c_value in enumerate(c_values):
        C.assign(c_value)
        u = diagnostic_solve_with_fallback(
            primary_solver,
            fallback_solver,
            velocity=u,
            thickness=thickness,
            surface=surface,
            stage_name=f"{stage_name} C={c_value:g} [{idx + 1}/{len(c_values)}]",
        )

    C.assign(target_c)
    return u


def run_simulation(
    primary_solver,
    fallback_solver,
    time,
    dt,
    *,
    thickness_inflow=None,
    stage_name="simulation",
    print_every=10,
    time_offset=0.0,
    **fields,
):
    h, s, u, z_b = map(
        fields.get, ("thickness", "surface", "velocity", "bed")
    )

    if h is None or s is None or u is None or z_b is None:
        raise ValueError(
            "run_simulation requires thickness, surface, velocity, and bed."
        )

    if thickness_inflow is None:
        thickness_inflow = h.copy(deepcopy=True)

    num_steps = int(round(time / dt))

    if num_steps <= 0:
        return {"thickness": h, "surface": s, "velocity": u}

    progress_bar = tqdm.trange(num_steps, desc=stage_name)

    for step in progress_bar:
        t0 = walltime.time()
        elapsed_years = time_offset + (step + 1) * dt
        assign_ramped_C(elapsed_years)

        h = primary_solver.prognostic_solve(
            dt,
            thickness=h,
            velocity=u,
            accumulation=a,
            thickness_inflow=thickness_inflow,
        )

        # Keep the thickness physically admissible.
        h.interpolate(max_value(h, 1.0))
        s = icepack.compute_surface(thickness=h, bed=z_b)

        u = diagnostic_solve_with_fallback(
            primary_solver,
            fallback_solver,
            velocity=u,
            thickness=h,
            surface=s,
            stage_name=f"{stage_name} step {step + 1}",
        )

        comm = h.function_space().mesh().comm
        local_min_h = float(h.dat.data_ro.min())
        min_h = comm.allreduce(local_min_h, op=MPI.MIN)
        avg_h = float(firedrake.assemble(h * dx) / DOMAIN_AREA)
        avg_speed = float(
            firedrake.assemble(firedrake.sqrt(firedrake.inner(u, u)) * dx) / DOMAIN_AREA
        )

        description = (
            f"{stage_name}: t={elapsed_years:g}, C={float(C):.3g}, "
            f"avg,min h={avg_h:4.2f},{min_h:4.2f}"
        )
        progress_bar.set_description(description)

        if (
            print_every is not None
            and (step % print_every == 0 or step == num_steps - 1)
        ):
            print(
                f"{stage_name} step {step + 1}/{num_steps}, t={elapsed_years:g} yr, "
                f"C={float(C):.6g}: avg h={avg_h:.3f}, min h={min_h:.3f}, "
                f"avg speed={avg_speed:.3f}, "
                f"wall={walltime.time() - t0:.2f} s",
                flush=True,
            )
            SPINUP_HISTORY.append(
                {
                    "stage": stage_name,
                    "step": step + 1,
                    "nsteps": num_steps,
                    "t_yr": float(elapsed_years),
                    "C": float(C),
                    "avg_h": avg_h,
                    "min_h": min_h,
                    "avg_speed": avg_speed,
                }
            )

    return {"thickness": h, "surface": s, "velocity": u}


def make_initial_fields(mesh, degree, *, source_fields=None):
    """
    If source_fields is None:
        use h = 100 m and the linear x velocity guess.
    If source_fields is provided:
        project source thickness and velocity to the new space.
    """
    Q = firedrake.FunctionSpace(mesh, "CG", degree)
    V = firedrake.VectorFunctionSpace(mesh, "CG", degree)

    z_b = Function(Q).interpolate(mismip_bed_topography(mesh))

    if source_fields is None:
        h_0 = Function(Q).assign(Constant(100))
        x = firedrake.SpatialCoordinate(mesh)[0]
        u_guess = Function(V).interpolate(as_vector((90 * x / Lx, 0)))
    else:
        h_0 = Function(Q).project(source_fields["thickness"])
        h_0.interpolate(max_value(h_0, 1.0))
        u_guess = Function(V).project(source_fields["velocity"])

    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

    assign_ramped_C(SIMULATION_ELAPSED_YEARS)
    icepack_solver = make_solver("icepack", monitor=False)
    petsc_solver = make_solver("petsc", monitor=MONITOR_SNES)

    u_0 = diagnostic_solve_robust(
        icepack_solver,
        petsc_solver,
        velocity=u_guess,
        thickness=h_0,
        surface=s_0,
        stage_name=f"CG{degree} initial diagnostic",
        friction_continuation=(
            source_fields is not None and abs(float(C) - float(C_START)) > 0.0
        ),
    )

    fields = {
        "surface": s_0,
        "thickness": h_0,
        "velocity": u_0,
    }

    thickness_inflow = h_0.copy(deepcopy=True)

    return fields, Q, V, z_b, thickness_inflow


def run_stage(
    stage_name,
    mesh,
    degree,
    total_time,
    dt,
    *,
    source_fields=None,
):
    """
    Run one CG-degree simulation stage with bidirectional solver fallback
    and log-linear C ramping from C_START to C_TARGET.
    C ramp time is measured cumulatively across stages (no reset).
    """
    global SIMULATION_ELAPSED_YEARS
    time_offset = SIMULATION_ELAPSED_YEARS
    print(
        f"\n=== {stage_name}: CG{degree}, total_time={total_time}, "
        f"dt={dt}, C: {C_START:g} -> {C_TARGET:g} over {C_RAMP_TIME:g} yr ===",
        flush=True,
    )

    fields, Q, V, z_b, thickness_inflow = make_initial_fields(
        mesh,
        degree,
        source_fields=source_fields,
    )

    primary_solver = make_solver(PRIMARY_SOLVER, monitor=MONITOR_SNES)
    fallback_kind = "icepack" if PRIMARY_SOLVER == "petsc" else "petsc"
    fallback_solver = make_solver(fallback_kind, monitor=False)

    fields = run_simulation(
        primary_solver,
        fallback_solver,
        total_time,
        dt,
        thickness_inflow=thickness_inflow,
        stage_name=stage_name,
        print_every=STEP_PRINT_EVERY,
        time_offset=time_offset,
        bed=z_b,
        **fields,
    )

    SIMULATION_ELAPSED_YEARS += float(total_time)
    return fields, Q, V, z_b, thickness_inflow


# ------------------------------------------------------------
# Mesh generation
# ------------------------------------------------------------

points = [
    (0, 0),
    (Lx, 0),
    (Lx, Ly),
    (0, Ly),
]

facets = [(i, (i + 1) % len(points)) for i in range(len(points))]
markers = list(range(1, len(points) + 1))

mesh_info = triangle.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets, facet_markers=markers)

dy = Ly / cfg["Ny"]
area = dy**2 / 2

triangle_mesh = triangle.build(mesh_info, max_volume=area)
coarse_mesh = icepack.meshing.triangle_to_firedrake(triangle_mesh)

save_mesh_plot(coarse_mesh, "00_initial_coarse_mesh.png", "Initial coarse mesh")


# ------------------------------------------------------------
# 1. Low resolution on coarse mesh: CG1
# ------------------------------------------------------------

fields_coarse_low, Q1_coarse, V1_coarse, z_b_coarse_low, h_inflow_coarse_low = (
    run_stage(
        "coarse_low",
        coarse_mesh,
        degree=1,
        total_time=COARSE_TOTAL_TIME,
        dt=COARSE_DT,
        source_fields=None,
    )
)

save_field_plot(
    fields_coarse_low["thickness"],
    "01_coarse_low_CG1_thickness.png",
    "Coarse mesh CG1 thickness",
)
save_field_plot(
    fields_coarse_low["velocity"],
    "01_coarse_low_CG1_velocity.png",
    "Coarse mesh CG1 velocity",
)


# ------------------------------------------------------------
# 2. High resolution on coarse mesh: CG2
# ------------------------------------------------------------

fields_coarse_high, Q2_coarse, V2_coarse, z_b_coarse_high, h_inflow_coarse_high = (
    run_stage(
        "coarse_high",
        coarse_mesh,
        degree=2,
        total_time=COARSE_TOTAL_TIME,
        dt=COARSE_DT,
        # Project CG1 spin-up onto CG2; avoids h=100 + target C mismatch after cumulative ramp.
        source_fields=fields_coarse_low,
    )
)

save_field_plot(
    fields_coarse_high["thickness"],
    "02_coarse_high_CG2_thickness.png",
    "Coarse mesh CG2 thickness",
)
save_field_plot(
    fields_coarse_high["velocity"],
    "02_coarse_high_CG2_velocity.png",
    "Coarse mesh CG2 velocity",
)



# ------------------------------------------------------------
# 3. Coarse-mesh error estimate: |CG2 - CG1|
# ------------------------------------------------------------

delta_h_coarse = Function(Q2_coarse).interpolate(
    abs(fields_coarse_high["thickness"] - fields_coarse_low["thickness"])
)

save_field_plot(
    delta_h_coarse,
    "03_coarse_delta_h_abs_CG2_minus_CG1.png",
    "Coarse mesh |CG2 - CG1| thickness discrepancy",
)


# ------------------------------------------------------------
# 4. Smooth the error into a DG0 field
# ------------------------------------------------------------

DG0_coarse = firedrake.FunctionSpace(coarse_mesh, "DG", 0)
epsilon = firedrake.Function(DG0_coarse)

J = 0.5 * (
    (epsilon - delta_h_coarse) ** 2 * dx
    + (Ly / 2) * (epsilon("+") - epsilon("-")) ** 2 * dS
)

F = firedrake.derivative(J, epsilon)
firedrake.solve(F == 0, epsilon)

save_field_plot(
    epsilon,
    "04_smoothed_error_DG0.png",
    "Smoothed DG0 thickness error indicator",
)


# ------------------------------------------------------------
# 5. Refine the Triangle mesh from the smoothed error
# ------------------------------------------------------------

def refine_triangle_mesh_from_error(
    triangle_mesh,
    firedrake_mesh,
    error_indicator,
    *,
    shrink=8,
    exponent=2,
):
    """
    This follows the notebook's Triangle refinement.
    """
    if firedrake_mesh.comm.size != 1:
        raise RuntimeError(
            "This MeshPy refinement block assumes one MPI rank. "
            "Run this adaptivity script serially, or add an explicit "
            "Triangle-cell to Firedrake-cell mapping before assigning "
            "element_volumes."
        )

    triangle_mesh.element_volumes.setup()

    DG0 = error_indicator.function_space()
    areas = firedrake.project(firedrake.CellVolume(firedrake_mesh), DG0)

    errors = error_indicator.dat.data_ro[:]
    cell_areas = areas.dat.data_ro[:]

    if len(errors) != len(triangle_mesh.elements):
        raise RuntimeError(
            f"DG0 error field has {len(errors)} cells, but Triangle mesh has "
            f"{len(triangle_mesh.elements)} elements. Cell ordering/numbering "
            "does not match."
        )

    max_err = float(errors.max()) if len(errors) > 0 else 0.0

    if max_err <= 0.0:
        print(
            "Warning: max error indicator is zero. Refinement will keep "
            "approximately the original areas.",
            flush=True,
        )
        max_err = 1.0

    for index, err in enumerate(errors):
        original_area = float(cell_areas[index])
        err_ratio = float(err) / max_err
        shrink_factor = shrink * err_ratio**exponent
        triangle_mesh.element_volumes[index] = original_area / (1.0 + shrink_factor)

    return triangle.refine(triangle_mesh)


refined_triangle_mesh = refine_triangle_mesh_from_error(
    triangle_mesh,
    coarse_mesh,
    epsilon,
    shrink=SHRINK,
    exponent=EXPONENT,
)

fine_mesh = icepack.meshing.triangle_to_firedrake(refined_triangle_mesh)

save_refined_mesh_plot(
    fine_mesh,
    fields_coarse_high,
    Q2_coarse,
    "05_refined_mesh_near_grounding_line.png",
)
save_mesh_plot(
    fine_mesh,
    "06_refined_mesh_full_domain.png",
    "Refined mesh",
)


# ------------------------------------------------------------
# 6. Low resolution on fine mesh: CG1
# ------------------------------------------------------------

fields_fine_low, Q1_fine, V1_fine, z_b_fine_low, h_inflow_fine_low = run_stage(
    "fine_low",
    fine_mesh,
    degree=1,
    total_time=FINE_TOTAL_TIME,
    dt=FINE_DT,
    source_fields=fields_coarse_high,
)

save_field_plot(
    fields_fine_low["thickness"],
    "07_fine_low_CG1_thickness.png",
    "Fine mesh CG1 thickness",
)
save_field_plot(
    fields_fine_low["velocity"],
    "07_fine_low_CG1_velocity.png",
    "Fine mesh CG1 velocity",
)


# ------------------------------------------------------------
# 7. High resolution on fine mesh: CG2
# ------------------------------------------------------------

fields_fine_high, Q2_fine, V2_fine, z_b_fine_high, h_inflow_fine_high = run_stage(
    "fine_high",
    fine_mesh,
    degree=2,
    total_time=FINE_TOTAL_TIME,
    dt=FINE_DT,
    source_fields=fields_coarse_high,
)

save_field_plot(
    fields_fine_high["thickness"],
    "08_fine_high_CG2_thickness.png",
    "Fine mesh CG2 thickness",
)
save_field_plot(
    fields_fine_high["velocity"],
    "08_fine_high_CG2_velocity.png",
    "Fine mesh CG2 velocity",
)



# ------------------------------------------------------------
# 8. Fine-mesh error estimate: |CG2 - CG1|
# ------------------------------------------------------------

delta_h_fine = Function(Q2_fine).interpolate(
    abs(fields_fine_high["thickness"] - fields_fine_low["thickness"])
)

save_field_plot(
    delta_h_fine,
    "09_fine_delta_h_abs_CG2_minus_CG1.png",
    "Fine mesh |CG2 - CG1| thickness discrepancy",
)

save_field_plot(
    fields_fine_high["thickness"],
    "10_final_fine_high_CG2_thickness.png",
    "Final fine mesh CG2 thickness",
)
save_field_plot(
    fields_fine_high["velocity"],
    "10_final_fine_high_CG2_velocity.png",
    "Final fine mesh CG2 velocity",
)
print(
    f"\nFinished adaptivity workflow. Figures written to: {OUTDIR}",
    flush=True,
)

# ------------------------------------------------------------
# 9. Save self-contained final steady state
# ------------------------------------------------------------

def to_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def checkpoint_paths(cfg):
    stem = cfg["output_stem"]
    save_dir = cfg["save_dir"]
    h5_path = os.path.join(save_dir, f"{stem}.h5")
    json_path = os.path.join(save_dir, f"{stem}.json")
    npz_path = os.path.join(save_dir, f"{stem}_grid.npz")
    return h5_path, json_path, npz_path


def save_cfg(cfg, json_path, h5_path=None):
    cfg_to_save = to_jsonable(cfg)
    with open(json_path, "w") as f:
        json.dump(cfg_to_save, f, indent=2, sort_keys=True)

    if h5_path is not None:
        try:
            import h5py

            with h5py.File(h5_path, "a") as h5:
                h5.attrs["cfg_json"] = json.dumps(cfg_to_save, sort_keys=True)
                h5.attrs["cfg_filename"] = os.path.basename(json_path)
        except Exception as err:
            print(
                f"Warning: could not embed cfg into HDF5 attributes: {repr(err)}",
                flush=True,
            )

    print(f"Saved cfg: {json_path}", flush=True)


def save_spinup_history(history, json_path):
    with open(json_path, "w") as f:
        json.dump({"points": history}, f, indent=2)
    print(f"Saved spin-up history: {json_path}", flush=True)


def evaluate_function_on_grid(function, pts, values, *, block=50_000):
    N = pts.shape[0]
    for i0 in range(0, N, block):
        i1 = min(i0 + block, N)
        p = pts[i0:i1]
        try:
            values[i0:i1] = function.at(p)
        except PointNotInDomainError:
            for j, point in enumerate(p):
                try:
                    values[i0 + j] = function.at(point)
                except PointNotInDomainError:
                    pass
    return values


def save_checkpoint_and_grid_npz(
    mesh,
    Q,
    V,
    fields,
    bed_field,
    cfg,
):
    h5_path, json_path, npz_path = checkpoint_paths(cfg)

    u = firedrake.Function(V, name="u")
    h = firedrake.Function(Q, name="h")
    s = firedrake.Function(Q, name="s")
    b = firedrake.Function(Q, name="b")

    A_field = firedrake.Function(Q, name="A")

    u.project(fields["velocity"])
    h.project(fields["thickness"])
    s.project(fields["surface"])
    b.project(bed_field)
    A_field.interpolate(Constant(cfg["A_field_saved_value"]))
    
    with firedrake.CheckpointFile(h5_path, "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(u, name="velocity")
        checkpoint.save_function(h, name="thickness")
        checkpoint.save_function(s, name="surface")
        checkpoint.save_function(b, name="bed")
        checkpoint.save_function(A_field, name="A")
        # checkpoint.save_function(eta_mpa_yr_func, name="eta_mpa_yr")

    print(f"Saved steady-state checkpoint: {h5_path}", flush=True)

    save_cfg(cfg, json_path, h5_path=h5_path)
    save_spinup_history(
        SPINUP_HISTORY,
        os.path.join(save_dir, f"{stem}_history.json"),
    )

    # Save the same final state on a regular 2D grid for non-Firedrake workflows.
    mesh_coords = mesh.coordinates.dat.data_ro
    xmin, ymin = mesh_coords.min(axis=0)
    xmax, ymax = mesh_coords.max(axis=0)

    resolution = float(cfg["grid_resolution"])
    x = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    y = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    x = x[x <= xmax + 1e-8]
    y = y[y <= ymax + 1e-8]

    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    N = pts.shape[0]

    h_vals = np.full(N, np.nan, dtype=float)
    s_vals = np.full(N, np.nan, dtype=float)
    b_vals = np.full(N, np.nan, dtype=float)
    A_vals = np.full(N, np.nan, dtype=float)
    speed_vals = np.full(N, np.nan, dtype=float)
    haf_vals = np.full(N, np.nan, dtype=float)
    viscosity_vals = np.full(N, np.nan, dtype=float)
    u_vals = np.full((N, 2), np.nan, dtype=float)

    speed = firedrake.Function(Q, name="speed").interpolate(sqrt(inner(u, u)))
    height_above_flotation = firedrake.Function(Q, name="height_above_flotation").interpolate(
        s - (1 - ρ_I / ρ_W) * h
    )
    # viscosity_field = firedrake.project(
    #     viscosity(velocity=u, thickness=h, fluidity=A),
    #     Q,
    # )
    eta_mpa_yr_func = firedrake.project(
        effective_viscosity( velocity=u,fluidity=A,),
        Q,
    )

    evaluate_function_on_grid(h, pts, h_vals)
    evaluate_function_on_grid(s, pts, s_vals)
    evaluate_function_on_grid(b, pts, b_vals)
    evaluate_function_on_grid(A_field, pts, A_vals)
    evaluate_function_on_grid(speed, pts, speed_vals)
    evaluate_function_on_grid(height_above_flotation, pts, haf_vals)
    evaluate_function_on_grid(eta_mpa_yr_func, pts, viscosity_vals)
    evaluate_function_on_grid(u, pts, u_vals)

    ny, nx = len(y), len(x)
    U = u_vals.reshape(ny, nx, 2)
    Ux = U[..., 0]
    Uy = U[..., 1]
    
    # Includes both descriptive names and the legacy names used by run_sim.py.
    np.savez_compressed(
        npz_path,
        x=x,
        y=y,
        X=X,
        Y=Y,
        h=h_vals.reshape(ny, nx),
        thickness=h_vals.reshape(ny, nx),
        s=s_vals.reshape(ny, nx),
        surface=s_vals.reshape(ny, nx),
        bed=b_vals.reshape(ny, nx),
        ux=Ux,
        uy=Uy,
        velocity=U,
        speed=speed_vals.reshape(ny, nx),
        A=A_vals.reshape(ny, nx),
        A_inv=A_vals.reshape(ny, nx),
        viscosity=viscosity_vals.reshape(ny, nx),
        height_above_flotation=haf_vals.reshape(ny, nx),
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        grid_resolution=resolution,
        cfg_json=json.dumps(to_jsonable(cfg), sort_keys=True),
    )

    print(
        f"Saved 2D gridded steady-state NPZ: {npz_path} "
        f"with shape (ny={ny}, nx={nx})",
        flush=True,
    )


save_checkpoint_and_grid_npz(
    fine_mesh,
    Q2_fine,
    V2_fine,
    fields_fine_high,
    z_b_fine_high,
    cfg,
)
