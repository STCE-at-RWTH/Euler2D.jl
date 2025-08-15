# Method skeleton for the exact Riemann solver.
# if NonlinearSolve.jl is available see <package root>/ext/ExactRiemannExt
# for actual implementation details
#
# the stubs are here to remind you to install NonlinearSolve.jl, which is a hefty dependency for a feature you may not need

# following the derviations in Smoller 1994
# Shock Waves and Reaction-Diffisuion Equations

module ExactRiemannSolverInternal

using LinearAlgebra
using StaticArrays

using Euler2D: dimensionless_pressure, dimensionless_speed_of_sound
using Euler2D: select_middle

function project_state_to_normal(u, n)
    ρv_n = select_middle(u) ⋅ n
    return SVector(u[1], ρv_n, u[end])
end

"""Pressure ratio ``P_r/P_l`` abbreviated pi"""
function _π(uL, uR, gas)
    return dimensionless_pressure(uR, gas) / dimensionless_pressure(uL, gas)
end

"""Density ratio ``ρ_r/ρ_l`` abbreviated z"""
_z(uL, uR) = uR[1] / uL[1]

"""Common coefficient computed from ``γ``"""
_β(gas) = (gas.γ + 1) / (gas.γ - 1)

"""Common coefficient computed from ``γ``"""
_τ(gas) = (gas.γ - 1) / (2 * gas.γ)

"""The parametrizations in Smoller rely on ``-\\log\\pi``"""
_x(uL, uR, gas) = -log(_π(uL, uR, gas))

"""
  density_bound_uR(uL, gas)

Compute and upper bound for the density of a post-shock state. I think.
"""
density_bound_uR(uL, gas) = (uL[1], _β(gas) * uL[1])

# TODO does the 1d theory work 1-1 here? as in, can I just ignore the shear wave...? Obviously I can just project down to a 1-d problem and work it out there.
# that might be the best plan.

"""
  jump_ratios(uL, uR, gas)

Compute the jump ratios:
```math
  A = ρ_r/ρ_l \\quad B = P_r/P_l \\quad C = (v_r - v_l) / c_l
```
"""
function jump_ratios(uL, uR, gas)
    A = _z(uL, uR)
    B = _π(uL, uR, gas)
    cL = dimensionless_speed_of_sound(uL, gas)
    C = (uR[2] / uR[1] - uL[2] / uL[1]) / cL
    return (A, B, C)
end

"""
  test_for_vacuum_in_solution(uL, uR, gas)

Theorem 18.6 in Smoller. Test if the Riemann problem between states ``u_L`` and ``u_R`` will yield a vacuum state, 
or if the solution will yield a solution composed of shocks, simple waves / rarefaction waves, and contact discontinuities.
"""
function test_for_vacuum_in_solution(uL, uR, gas)
    (cL, cR) = dimensionless_speed_of_sound.((uL, uR), gas)
    vL = uL[2] / uL[1]
    vR = uR[2] / uR[1]
    return vR - vL ≥ 2 / (gas.γ - 1) * (cL + cR)
end

function f_1(x, gas)
    return if x > 0
        # shimple wave
        exp(-x / gas.γ)
    else
        # shock
        (_β(gas) + exp(x)) / (1 + _β(gas) * exp(x))
    end
end

function f_3(x, gas)
    return 1.0 / f_1(x, gas)
end

function h_1(x, gas)
    return if x > 0
        # simple wave
        2 / (gas.γ - 1) * (1 - exp(-_τ(gas) * x))
    else
        # shock
        c1 = 2 * sqrt(_τ(gas)) / (gas.γ - 1)
        c1 * (1 - exp(-x)) / sqrt(1 + _β(gas) * exp(-x))
    end
end

function h_3(x, gas)
    return sqrt(f_1(x, gas)) * exp(x / 2) * h_1(x, gas)
end

"""
  test_1_wave_for_rarefaction(uL, uR, gas)

Corallary 18.7 in Smoller. Test if the 1-wave in the solution to the Riemann problem between ``u_L`` and ``u_R`` is a rarefaction wave.
"""
function test_1_wave_for_rarefaction(uL, uR, gas)
    (A, B, C) = jump_ratios(uL, uR, gas)
    c1 = sqrt(B / A)
    lhs = c1 * h_1(log(B), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < C < rhs
end

"""
  test_3_wave_for_rarefaction(uL, uR, gas)

Corallary 18.7 in Smoller. Test if the 3-wave in the solution to the Riemann problem between ``u_L`` and ``u_R`` is a rarefaction wave.
"""
function test_3_wave_for_rarefaction(uL, uR, gas)
    (A, B, C) = jump_ratios(uL, uR, gas)
    c1 = sqrt(B / A)
    lhs = h_1(-log(B), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < C < rhs
end

"""
  solve_for_jump_parameters(uL, uR, gas, n = 1)

Solve for the jump parameters ``(x_1, x_2, x_3)`` as outlined in Smoller.

Will warn and do nothing if the ExactRiemannExt is not loaded.
"""
function solve_for_jump_parameters(args...)
    @warn "NonlinearSolve not available or wrong arguments."
    nothing
end

function solve_riemann_problem(ray, uL, uR, gas, n)
    x1, x2, x3 = solve_for_jump_parameters(uL, uR, gas, n)
end

end

function ϕ_exact(uL, uR, n, gas)
    u_riemann = ExactRiemannSolverInternal.solve_riemann_problem(0, uL, uR, n, gas)

    return F_euler(u_riemann, gas)
end
