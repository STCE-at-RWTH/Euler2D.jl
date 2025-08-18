# Method skeleton for the exact Riemann solver.
# if NonlinearSolve.jl is available see <package root>/ext/ExactRiemannExt
# for actual implementation details
#
# the stubs are here to remind you to install NonlinearSolve.jl, which is a hefty dependency for a feature you may not need

# following the derviations in Smoller 1994
# Shock Waves and Reaction-Diffisuion Equations

module ExactRiemannSolverInternal

using LinearAlgebra
using SimpleNonlinearSolve
using StaticArrays

using Euler2D: dimensionless_pressure, dimensionless_speed_of_sound
using Euler2D: select_middle, project_state_to_normal

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

# Defined below lemma 18.5

function T_1(ρ, P, v, x, gas)
    return (f_1(x, gas) * ρ, exp(-x) * P, v + sqrt(gas.γ * P / ρ) * h_1(x, gas))
end

function T_2(ρ, P, v, x, gas)
    return (exp(x) * ρ, P, v)
end

function T_3(ρ, P, v, x, gas)
    return (f_3(x, gas) * ρ, exp(x) * P, v + sqrt(gas.γ * P / ρ) * h_3(x, gas))
end

"""
  test_1_wave_for_rarefaction(uL, uR, gas)

Corallary 18.7 in Smoller. Test if the 1-wave in the solution to the Riemann problem between ``u_L`` and ``u_R`` is a rarefaction wave.
"""
function test_1_wave_for_rarefaction(uL, uR, gas)
    return test_1_wave_for_rarefaction(jump_ratios(uL, uR, gas), gas)
end

function test_1_wave_for_rarefaction(jumps, gas)
    (A, B, C) = jumps
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
    return test_3_wave_for_rarefaction(jump_ratios(uL, uR, gas), gas)
end

function test_3_wave_for_rarefaction(jumps, gas)
    (A, B, C) = jumps
    c1 = sqrt(B / A)
    lhs = h_1(-log(B), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < C < rhs
end

"""
  solve_for_jump_parameters(uL, uR, gas, n = 1)

Solve for the jump parameters ``(x_1, x_2, x_3)`` as outlined in Smoller.
"""
function solve_for_jump_parameters(state_left, state_right, gas, n = 1)
    uL = project_state_to_normal(state_left, n)
    uR = project_state_to_normal(state_right, n)

    p = SVector(jump_ratios(uL, uR, gas)...)
    nonlinear_system = NonlinearProblem(ones(SVector{eltype(uL),3}), p) do x, params
        # 18.62 stuff - A = 0
        v1 = f_1(x[1], gas) * exp(x[2]) * f_3(x[3], gas) - params[1]
        # 18.61: stuff - B = 0 
        v2 = exp(x[3] - x[1]) - params[2]
        # 18.63: stuff - C = 0
        v3 = h_1(x[1], gas) + sqrt(p[2] / p[1]) * h_1(x[1] + log(p[2]), gas) - params[3]
        return SVector(v1, v2, v3)
    end
    solution = solve(nonlinear_system)
    return solution.u
end

function to_ρPv(u, gas)
    return (u[1], dimensionless_pressure(u, gas), u[2] / u[1])
end

function solve_riemann_problem(ray, uL, uR, gas, n)
    if test_for_vacuum_in_solution(uL, uR, gas)
        throw(DomainError((uL, uR), "Vacuum state forms between uL and uR!"))
    end
    x1, x2, x3 = solve_for_jump_parameters(uL, uR, gas, n)
    uL_prim = to_ρPv(uL, gas)
    u1_prim = T_1(uL_prim..., x1, gas)
    u2_prim = T_2(u1_prim..., x2, gas)
    u3_prim = T_3(u2_prim..., x3, gas)

    #TODO u3_prim should be close to uR_prim... I think.
    #TODO pick which state
    #TODO add the off-dimension components back in
    #TODO convert back to "real" co-ordinates
end

end

function ϕ_exact(uL, uR, n, gas)
    u_riemann = ExactRiemannSolverInternal.solve_riemann_problem(0, uL, uR, n, gas)
    return F_euler(u_riemann, gas)
end
