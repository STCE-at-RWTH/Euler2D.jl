# Method skeleton for the exact Riemann solver.
# following the derviations in Smoller 1994
# Shock Waves and Reaction-Diffisuion Equations

module ExactRiemannSolverInternal

using LinearAlgebra
using SimpleNonlinearSolve
using StaticArrays

using Euler2D: dimensionless_pressure, dimensionless_speed_of_sound, F_euler
using Euler2D: select_middle, apply_coordinate_tform, change_of_basis_matrix

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
    cL = dimensionless_speed_of_sound(uL, gas)
    cR = dimensionless_speed_of_sound(uR, gas)
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
    return (ρ = f_1(x, gas) * ρ, P = exp(-x) * P, v = v + sqrt(gas.γ * P / ρ) * h_1(x, gas))
end

function T_2(ρ, P, v, x, gas)
    return (ρ = exp(x) * ρ, P = P, v = v)
end

function T_3(ρ, P, v, x, gas)
    return (ρ = f_3(x, gas) * ρ, P = exp(x) * P, v = v + sqrt(gas.γ * P / ρ) * h_3(x, gas))
end

"""
  test_1_wave_for_rarefaction(uL, uR, gas)

Corallary 18.7 in Smoller. Test if the 1-wave in the solution to the Riemann problem between ``q_L`` and ``q_R`` is a rarefaction wave.
The states ``q`` are already projected into interface-normal co-ordinates.
"""
function test_1_wave_for_rarefaction(qL, qR, gas)
    return test_1_wave_for_rarefaction(jump_ratios(qL, qR, gas), gas)
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

Corallary 18.7 in Smoller. Test if the 3-wave in the solution to the Riemann problem between ``q_L`` and ``q_R`` is a rarefaction wave.
The states ``q`` are already projected into interface-normal co-ordinates.
"""
function test_3_wave_for_rarefaction(qL, qR, gas)
    return test_3_wave_for_rarefaction(jump_ratios(qL, qR, gas), gas)
end

function test_3_wave_for_rarefaction(jumps, gas)
    (A, B, C) = jumps
    c1 = sqrt(B / A)
    lhs = h_1(-log(B), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < C < rhs
end

"""
  solve_for_jump_parameters(qL, qR, gas)

Solve for the jump parameters ``(x_1, x_2, x_3)`` as outlined in Smoller.
- `qL` and `qR` are already projected into the interface-normal co-ordinates.
"""
function solve_for_jump_parameters(qL, qR, gas)
    jump_params = SVector(jump_ratios(qL, qR, gas)...)
    x0 = @SVector ones(eltype(qL), 3)
    nonlinear_system = NonlinearProblem(x0, jump_params) do x, params
        # 18.62 stuff - A = 0
        v1 = f_1(x[1], gas) * exp(x[2]) * f_3(x[3], gas) - params[1]
        # 18.61: stuff - B = 0 
        v2 = exp(x[3] - x[1]) - params[2]
        # 18.63: stuff - C = 0
        v3 =
            h_1(x[1], gas) +
            sqrt(jump_params[2] / jump_params[1]) * h_1(x[1] + log(jump_params[2]), gas) - params[3]
        return SVector(v1, v2, v3)
    end
    solution = solve(nonlinear_system, SimpleNewtonRaphson())
    return solution.u
end

function to_ρPv(q, gas)
    return (ρ = q[1], P = dimensionless_pressure(q, gas), v = select_middle(q) ./ q[1])
end

sound_speed(state_prim, gas) = sqrt(gas.γ * state_prim.P / state_prim.ρ)

function to_conserved(ρ, P, v, v_shear, gas)
    v_all = pushfirst(v_shear, v)
    ρv_all = ρ * v_all
    ρE = P / (gas.γ - 1) + 0.5 * (ρv_all ⋅ v_all)
    return SVector(ρ, ρv_all..., ρE)
end

"""
Find shock speed for two states `uL`, `uR` connected by a shock.
"""
function shock_speed_via_RH(uL, uR, gas)
    fL = F_euler(uL, gas)[:, 1]
    fR = F_euler(uR, gas)[:, 1]
    f_jump = fR - fL
    u_jump = uR - uL
    # we know that f_jump must be a scalar multiple of u_jump
    # and in fact u_jump is an eigenvector of f'(u) in the small disturbance limit
    return f_jump[1] / u_jump[1]
end

"""
Find rarefaction structure for states connected to state `L` by a 1-rarefaction. 
"""
function rarefaction_1(ray, prim_L, cL, gas)
    u = ((gas.γ - 1.0) * first(prim_L.v) + 2 * (cL + ray)) / (gas.γ + 1.0)
    ρ = ((prim_L.ρ^gas.γ * (u - ray)^2) / (gas.γ * prim_L.P))^(1 / (gas.γ - 1))
    P = prim_L.P * (ρ / prim_L.ρ)^gas.γ
    return (ρ = ρ, P = P, v = u)
end

"""
Find rarefaction structure for states connected to state `R` by a 3-rarefaction. 
"""
function rarefaction_3(ray, prim_R, cR, gas)
    u = ((gas.γ - 1.0) * first(prim_R.v) - 2 * (cR - ray)) / (gas.γ + 1.0)
    ρ = ((prim_R.ρ^gas.γ * (ray - u)^2) / (gas.γ * prim_R.P))^(1 / (gas.γ - 1))
    P = prim_R.P * (ρ / prim_R.ρ)^gas.γ
    return (ρ = ρ, P = P, v = u)
end

function riemann_states_and_speeds(qL, qR, gas)
    x1, x2, x3 = solve_for_jump_parameters(qL, qR, gas)
    one_shock = x1 < 0
    three_shock = x3 < 0

    # switch to ρ, P, v
    qL_prim = to_ρPv(qL, gas)
    qR_prim = to_ρPv(qR, gas)
    # apply the transformation functions to find the states between qL and qR
    # that are separated by a contact wave
    qL_prim_★ = T_1(qL_prim.ρ, qL_prim.P, first(qL_prim.v), x1, gas)
    qR_prim_★ = T_2(values(qL_prim_★)..., x2, gas)
    @assert isapprox(qL_prim_★[3], qR_prim_★[3]; atol = 1.0e-8) # assert velocity constant across contact wave!
    # contact discontinuitity propagates at this speed
    contact_speed = qL_prim_★.v
    @assert isapprox(qL_prim_★[2], qR_prim_★[2]; atol = 1.0e-8) # assert pressure constant across contact wave!
    # qR_prim_appx = T_3(qR_prim_★..., x3, gas)

    cL = dimensionless_speed_of_sound(qL, gas)
    cR = dimensionless_speed_of_sound(qR, gas)
    cL_★ = sound_speed(qL_prim_★, gas)
    cR_★ = sound_speed(qR_prim_★, gas)

    speeds_1 = if one_shock
        # solve just the first component of RH
        s = (qL_prim_★.ρ * qL_prim_★.v - qL[2]) / (qL_prim_★.ρ - qL[1])
        (s, s)
    else
        (first(qL_prim.v) - cL, contact_speed - cL_★)
    end

    speeds_3 = if three_shock
        s = (qR[2] - qR_prim_★.ρ * qR_prim_★.v) / (qR[2] - qR_prim_★.ρ)
        (s, s)
    else
        (contact_speed + cR_★, first(qR_prim.v) + cR)
    end
    return (
        one_shock,
        speeds_1,
        qL_prim,
        qL_prim_★,
        contact_speed,
        three_shock,
        speeds_3,
        qR_prim_★,
        qR_prim,
    )
end

function solve_riemann_problem(ray, uL, uR, gas, new_coords)
    to_normal_coords = change_of_basis_matrix(I, new_coords)
    qL = apply_coordinate_tform(uL, to_normal_coords)
    qR = apply_coordinate_tform(uR, to_normal_coords)
    (
        has_one_shock,
        (sL_1, sL_2),
        qL_prim,
        qL_prim_★,
        contact_speed,
        has_three_shock,
        (sR_1, sR_2),
        qR_prim_★,
        qR_prim,
    ) = riemann_states_and_speeds(qL, qR, gas)

    shear_v_L = popfirst(qL_prim.v)
    shear_v_R = popfirst(qR_prim.v)
    if ray < contact_speed
        # 1-family
        if has_one_shock
            if ray < sL_1
                return uL
            end
            state = to_conserved(values(qL_prim_★)..., shear_v_L, gas)
            return apply_coordinate_tform(state, new_coords)
        else
            if ray < sL_1
                return uL
            elseif ray < sL_2
                raref_state = rarefaction_1(ray, qL_prim, sound_speed(qL_prim, gas), gas)
                state = to_conserved(values(raref_state)..., shear_v_L, gas)
                return apply_coordinate_tform(state, new_coords)
            else
                state = to_conserved(values(qL_prim_★)..., shear_v_L, gas)
                return apply_coordinate_tform(state, new_coords)
            end
        end
    else
        # 3-family
        if has_three_shock
            if ray > sR_2
                return uR
            end
            state = to_conserved(values(qR_prim_★)..., shear_v_R, gas)
            return apply_coordinate_tform(state, new_coords)
        else
            if ray < sR_1
                state = to_conserved(values(qR_prim_★)..., shear_v_R, gas)
                return apply_coordinate_tform(state, new_coords)
            elseif ray < sR_2
                raref_state = rarefaction_3(ray, qR_prim, sound_speed(qR_prim, gas), gas)
                state = to_conserved(values(raref_state)..., shear_v_R, gas)
                return apply_coordinate_tform(state, new_coords)
            else
                return uR
            end
        end
    end
end

end

"""
  ϕ_exact(uL, uR, n, gas)

Compute the value of ``F(u)\\cdot n`` on a boundary with unit normal ``n`` using an exact Riemann solver. 

"""
function ϕ_exact(uL, uR, n, gas)
    coords = orthonormal_basis(n)
    u_riemann = ExactRiemannSolverInternal.solve_riemann_problem(0, uL, uR, coords, gas)
    return F_euler(u_riemann, gas) * n
end
