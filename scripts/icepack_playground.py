#!/usr/bin/env python3
"""Hands-on icepack exercises — run pieces to learn the platform.

Usage (with firedrake env active):
    python icepack_playground.py          # run all quick exercises
    python icepack_playground.py 1 3 5    # run selected exercises

Official tutorials live at:
    ~/.firedrake-conda/clones/icepack/notebooks/tutorials/
"""

from __future__ import annotations

import sys

import firedrake
from firedrake import Function, SpatialCoordinate, as_vector, Constant, dx, assemble
import icepack
import icepack.plot
import numpy as np


def exercise_1_meshes_and_plotting():
    """Firedrake meshes, boundary IDs, symbolic fields, icepack plotting."""
    print("\n=== 1. Meshes & plotting ===")
    mesh = firedrake.UnitSquareMesh(16, 16)
    x, y = SpatialCoordinate(mesh)

    # Symbolic thickness field (parabolic dome)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    h = Function(Q).interpolate(500 * (1 - (x - 0.5) ** 2 - (y - 0.5) ** 2))

    volume = assemble(h * dx)
    print(f"  Integrated thickness (m³ per unit width): {float(volume):.1f}")
    print(f"  Boundary segments: {mesh.exterior_facets.unique_markers}")

    # Uncomment in Jupyter to display (plotting is in firedrake, not icepack.plot):
    # fig, axes = icepack.plot.subplots()
    # firedrake.triplot(mesh, axes=axes)
    # firedrake.tricontourf(h, axes=axes)


def exercise_2_ice_shelf_diagnostic():
    """Floating ice shelf: solve for velocity, compare to analytical profile."""
    print("\n=== 2. Ice shelf diagnostic solve ===")
    from icepack.constants import ice_density as rho_I, water_density as rho_W, gravity as g, glen_flow_law as n

    Lx, Ly = 20e3, 20e3
    u0, h0, dh, T = 100.0, 500.0, 100.0, 254.15

    def exact_u(x):
        rho = rho_I * (1 - rho_I / rho_W)
        h = h0 - dh * x / Lx
        P = rho * g * h / 4
        P0 = rho * g * h0 / 4
        dP = rho * g * dh / 4
        A = icepack.rate_factor(T)
        return u0 + Lx * A * (P0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * dP)

    mesh = firedrake.RectangleMesh(24, 24, Lx, Ly)
    x, y = SpatialCoordinate(mesh)
    degree = 2
    V = firedrake.VectorFunctionSpace(mesh, "CG", degree)
    Q = firedrake.FunctionSpace(mesh, "CG", degree)

    u_exact = Function(V).interpolate(as_vector((exact_u(x), 0)))
    h = Function(Q).interpolate(h0 - dh * x / Lx)
    A = Function(Q).assign(Constant(icepack.rate_factor(T)))

    model = icepack.models.IceShelf()
    solver = icepack.solvers.FlowSolver(
        model, dirichlet_ids=[1], side_wall_ids=[3, 4]
    )
    u = solver.diagnostic_solve(
        velocity=Function(V).interpolate(u_exact),
        thickness=h,
        fluidity=A,
        strain_rate_min=Constant(0.0),
    )

    rel_err = icepack.norm(u_exact - u, norm_type="H1") / icepack.norm(u_exact, norm_type="H1")
    print(f"  Relative H1 error vs analytical: {float(rel_err):.2e}")
    print(f"  Max speed (m/yr): {float(u.dat.data.max()):.1f}")


def exercise_3_ice_stream():
    """Grounded ice stream with basal sliding — compare to manufactured solution."""
    print("\n=== 3. Ice stream ===")
    from icepack.constants import (
        ice_density as rho_I,
        water_density as rho_W,
        glen_flow_law as n,
        weertman_sliding_law as m,
        gravity as g,
    )

    Lx, Ly = 20e3, 20e3
    h0, dh, s0, ds = 500.0, 100.0, 150.0, 90.0
    T = 254.15
    u_inflow = 100.0
    h_L, s_L = h0 - dh, s0 - ds
    beta = dh / ds * (rho_I * h_L**2 - rho_W * (s_L - h_L) ** 2) / (rho_I * h_L**2)

    def exact_u(x):
        A = icepack.rate_factor(T)
        rho = beta * rho_I * ds / dh
        h = h0 - dh * x / Lx
        P = rho * g * h / 4
        dP = rho * g * dh / 4
        P0 = rho * g * h0 / 4
        return u_inflow + Lx * A * (P0 ** (n + 1) - P ** (n + 1)) / ((n + 1) * dP)

    def friction(x):
        h = h0 - dh * x / Lx
        return (1 - beta) * (rho_I * g * h) * ds / Lx * exact_u(x) ** (-1 / m)

    mesh = firedrake.RectangleMesh(24, 24, Lx, Ly)
    x, y = SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, "CG", 2)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 2)

    u_exact = Function(V).interpolate(as_vector((exact_u(x), 0)))
    h = Function(Q).interpolate(h0 - dh * x / Lx)
    s = Function(Q).interpolate(s0 - ds * x / Lx)
    C = Function(Q).interpolate(friction(x))
    A = Function(Q).assign(Constant(icepack.rate_factor(T)))

    solver = icepack.solvers.FlowSolver(
        icepack.models.IceStream(), dirichlet_ids=[1], side_wall_ids=[3, 4]
    )
    u = solver.diagnostic_solve(
        velocity=Function(V).interpolate(u_exact),
        thickness=h,
        surface=s,
        fluidity=A,
        friction=C,
    )

    rel_err = icepack.norm(u_exact - u, norm_type="H1") / icepack.norm(u_exact, norm_type="H1")
    print(f"  Relative H1 error vs analytical: {float(rel_err):.2e}")
    speed = Function(Q).interpolate(firedrake.sqrt(firedrake.dot(u, u)))
    print(f"  Max speed (m/yr): {float(speed.dat.data.max()):.1f}")


def exercise_4_shallow_ice_disk():
    """Shallow-ice approximation on a circular domain (Bueler profile)."""
    print("\n=== 4. Shallow ice (Bueler disk) ===")
    from firedrake import grad, inner, sqrt, max_value, UnitDiskMesh
    from icepack.constants import ice_density as rho_I, glen_flow_law as n, gravity as g

    R = 100e3
    mesh = UnitDiskMesh(3)
    mesh.coordinates.dat.data[:] *= 0.75 * R
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x**2 + y**2)

    T = Constant(254.15)
    A = icepack.rate_factor(T)
    A0 = 2 * A * (rho_I * g) ** n / (n + 2)
    alpha = Constant(R)
    h_divide = (2 * R * (alpha / A0) ** (1 / n) * (n - 1) / n) ** (n / (2 * n + 2))
    h_part2 = (
        (n + 1) * (r / R)
        - n * (r / R) ** ((n + 1) / n)
        + n * max_value(1 - r / R, 0) ** ((n + 1) / n)
        - 1
    )
    h_expr = (h_divide / ((n - 1) ** (n / (2 * n + 2)))) * max_value(h_part2, 0) ** (
        n / (2 * n + 2)
    )

    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
    h = Function(Q).interpolate(h_expr)
    s = Function(Q).interpolate(h_expr)
    u_exact = Function(V).interpolate(-A0 * h ** (n + 1) * inner(grad(h), grad(h)) * grad(h))

    model = icepack.models.ShallowIce()
    u = icepack.solvers.FlowSolver(model).diagnostic_solve(
        velocity=Function(V), thickness=h, surface=s, fluidity=Constant(A)
    )

    rel_err = icepack.norm(u_exact - u, norm_type="L2") / icepack.norm(u_exact, norm_type="L2")
    print(f"  Max thickness (m): {float(h.dat.data.max()):.1f}")
    print(f"  Relative L2 error: {float(rel_err):.2e}")


def exercise_5_mass_transport_step():
    """Advect thickness with constant velocity (prognostic step)."""
    print("\n=== 5. Mass transport (one step) ===")
    Lx, Ly = 1.0, 1.0
    mesh = firedrake.RectangleMesh(32, 32, Lx, Ly)
    x, y = SpatialCoordinate(mesh)
    Q = firedrake.FunctionSpace(mesh, "CG", 1)
    V = firedrake.VectorFunctionSpace(mesh, "CG", 1)

    h_in, dh, u0 = 1.0, 0.2, 1.0
    h = Function(Q).interpolate(h_in - dh * x / Lx)
    h_inflow = Function(Q).interpolate(h_in - dh * x / Lx)
    u = Function(V).interpolate(as_vector((u0, 0)))
    a = Function(Q).assign(Constant(0.0))

    solver = icepack.solvers.FlowSolver(
        icepack.models.IceShelf(), prognostic_solver_type="lax-wendroff"
    )
    dt = 0.05
    h_new = solver.prognostic_solve(
        dt, thickness=h, velocity=u, accumulation=a, thickness_inflow=h_inflow
    )

    z = x - u0 * dt
    h_expected = Function(Q).interpolate(h_in - dh / Lx * firedrake.max_value(0, z))
    change = icepack.norm(h_new - h_expected, norm_type="L1") / icepack.norm(h_expected, norm_type="L1")
    print(f"  Relative L1 error after one step: {float(change):.4f}")


def exercise_6_material_properties():
    """Explore rate factor, Glen law, and constants."""
    print("\n=== 6. Material properties ===")
    temps = [250.0, 254.15, 260.0, 268.0]
    for T in temps:
        A = icepack.rate_factor(T)
        print(f"  T={T:5.1f} K  rate_factor A = {A:.3e} Pa^-3 yr^-1")
    print(f"  Glen n = {icepack.constants.glen_flow_law}")
    print(f"  rho_ice = {icepack.constants.ice_density} kg/m³")


def exercise_7_hybrid_model_smoke():
    """2D flowband hybrid model (xz extruded mesh)."""
    print("\n=== 7. Hybrid model (flowband) ===")
    Lx = 20e3
    mesh_x = firedrake.IntervalMesh(24, Lx)
    mesh = firedrake.ExtrudedMesh(mesh_x, layers=4, layer_height=1.0)
    x, zeta = SpatialCoordinate(mesh)

    h0, dh, ds, d = 500.0, 100.0, 90.0, 50.0
    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="DG", vdegree=0)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2)

    h = Function(Q).interpolate(h0 - dh * x / Lx)
    s = Function(Q).interpolate(d + h0 - dh + ds * (1 - x / Lx))
    A = Constant(icepack.rate_factor(254.15))
    C = Constant(0.001)
    u0 = Function(V).interpolate((0.95 + 0.05 * zeta) * (100.0 + 50.0 * x / Lx))

    u = icepack.solvers.FlowSolver(
        icepack.models.HybridModel(), dirichlet_ids=[1], tol=1e-10
    ).diagnostic_solve(
        velocity=u0, thickness=h, surface=s, fluidity=A, friction=C
    )
    speed = Function(Q).interpolate(firedrake.abs(u))
    print(f"  Hybrid solve OK — max |u| = {float(speed.dat.data.max()):.1f} m/yr")


EXERCISES = {
    1: exercise_1_meshes_and_plotting,
    2: exercise_2_ice_shelf_diagnostic,
    3: exercise_3_ice_stream,
    4: exercise_4_shallow_ice_disk,
    5: exercise_5_mass_transport_step,
    6: exercise_6_material_properties,
    7: exercise_7_hybrid_model_smoke,
}


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        keys = [int(k) for k in argv[1:]]
    else:
        keys = list(EXERCISES)

    print("icepack playground — make sure conda env ~/firedrake-env is active")
    for k in keys:
        if k not in EXERCISES:
            print(f"Unknown exercise {k}. Choose from {list(EXERCISES)}")
            return 1
        EXERCISES[k]()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
