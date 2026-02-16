# Method skeleton for the exact Riemann solver.
# following the derviations in Smoller 1994
# Shock Waves and Reaction-Diffisuion Equations

module ExactRiemannSolver

using LinearAlgebra
using SimpleNonlinearSolve
using StaticArrays

using PlanePolygons: change_of_basis_matrix, orthonormal_basis
using Euler2D: Euler2D
using Euler2D: dimensionless_velocity, dimensionless_pressure
using Euler2D: dimensionless_speed_of_sound, F_euler
using Euler2D: select_middle, change_space_coordinates

"""Pressure ratio ``P_r/P_l`` abbreviated ``π``"""
function _π(uL, uR, gas)
    return dimensionless_pressure(uR, gas) / dimensionless_pressure(uL, gas)
end

"""Density ratio ``ρ_r/ρ_l`` abbreviated ``z``"""
_z(uL, uR) = first(uR) / first(uL)

"""Common coefficient computed from ``γ``"""
_β(gas) = (gas.γ + 1) / (gas.γ - 1)

"""Common coefficient computed from ``γ``"""
_τ(gas) = (gas.γ - 1) / (2 * gas.γ)

"""The parametrizations in Smoller rely on ``-log π``"""
_x(uL, uR, gas) = -log(_π(uL, uR, gas))

"""
  density_bound_uR(uL, gas)

Compute and upper bound for the density of a post-shock state. I think.
"""
density_bound_uR(uL, gas) = (uL[1], _β(gas) * uL[1])

"""
    JumpRatios{T}

# Fields

- `Q_density` = ``ρ_R / ρ_L``
- `Q_pressure` = ``P_R / P_L``
- `Q_Δv_c` = ``(v_R - v_L) / c_L`` (where ``c_L`` is the speed of sound on the *LEFT*)

# Methods
These three are defined in **18.60** in Smoller.
- [`smoller_riemann_A(::JumpRatios)`](@ref)
- [`smoller_riemann_B(::JumpRatios)`](@ref)
- [`smoller_riemann_C(::JumpRatios)`](@ref)
"""
struct JumpRatios{T}
    # Q for Quotient!
    Q_density::T
    Q_pressure::T
    Q_Δv_c::T
end

# 18.60 from Smoller
smoller_riemann_A(ratios) = ratios.Q_density
smoller_riemann_B(ratios) = ratios.Q_pressure
smoller_riemann_C(ratios) = ratios.Q_Δv_c

function JumpRatios(uL::AbstractVector, uR::AbstractVector, gas)
    A = _z(uL, uR)
    B = _π(uL, uR, gas)
    Δv1 = first(dimensionless_velocity(uR)) - first(dimensionless_velocity(uL))
    C = Δv1 / dimensionless_speed_of_sound(uL, gas)
    return JumpRatios(A, B, C)
end

"""
    riemann_problem_has_solution(uL, uR, gas)
    test_for_vacuum_in_solution(uL, uR, gas) 

Theorem 18.6 in Smoller. Test if the Riemann problem between states ``u_L`` and ``u_R`` will yield a vacuum state, 
or if the solution will yield a solution composed of shocks, simple waves / rarefaction waves, and contact discontinuities.

See also: [`test_for_vacuum_in_solution`](@ref)
"""
function riemann_problem_has_solution(uL, uR, gas)
    cL = dimensionless_speed_of_sound(uL, gas)
    cR = dimensionless_speed_of_sound(uR, gas)
    vL = uL[2] / uL[1]
    vR = uR[2] / uR[1]
    return vR - vL < 2 / (gas.γ - 1) * (cL + cR)
end

function test_for_vacuum_in_solution(uL, uR, gas)
    return !riemann_problem_has_solution(uL, uR, gas)
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
    return (
        ρ = f_1(x, gas) * ρ,
        P = exp(-x) * P,
        v = only(v) + sqrt(gas.γ * P / ρ) * h_1(x, gas),
    )
end
T_1(u_prim::NamedTuple{(:ρ, :P, :v)}, x, gas) = T_1(u_prim.ρ, u_prim.P, u_prim.v, x, gas)

function T_2(ρ, P, v, x)
    return (ρ = exp(x) * ρ, P = P, v = only(v))
end
T_2(u_prim::NamedTuple{(:ρ, :P, :v)}, x) = T_2(u_prim.ρ, u_prim.P, u_prim.v, x)

function T_3(ρ, P, v, x, gas)
    return (
        ρ = f_3(x, gas) * ρ,
        P = exp(x) * P,
        v = only(v) + sqrt(gas.γ * P / ρ) * h_3(x, gas),
    )
end
T_3(u_prim::NamedTuple{(:ρ, :P, :v)}, x, gas) = T_3(u_prim.ρ, u_prim.P, u_prim.v, x, gas)

"""
    test_1_wave_for_rarefaction(qL, qR, gas)

Corallary 18.7 in Smoller. Test if the 1-wave in the solution to the Riemann problem between ``q_L`` and ``q_R`` is a rarefaction wave.

Note: The states ``q_{L, R}`` are already projected into interface-normal co-ordinates.
"""
function test_1_wave_for_rarefaction(qL, qR, gas)
    return test_1_wave_for_rarefaction(JumpRatios(qL, qR, gas), gas)
end

function test_1_wave_for_rarefaction(jump_ratios, gas)
    c1 = sqrt(smoller_riemann_B(jump_ratios) / smoller_riemann_A(jump_ratios))
    lhs = c1 * h_1(log(smoller_riemann_B(jump_ratios)), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < smoller_riemann_C(jump_ratios) < rhs
end

"""
    test_3_wave_for_rarefaction(uL, uR, gas)

Corallary 18.7 in Smoller. Test if the 3-wave in the solution to the Riemann problem between ``q_L`` and ``q_R`` is a rarefaction wave.
The states ``q`` are already projected into interface-normal co-ordinates.
"""
function test_3_wave_for_rarefaction(qL, qR, gas)
    return test_3_wave_for_rarefaction(JumpRatios(qL, qR, gas), gas)
end

function test_3_wave_for_rarefaction(jumps, gas)
    c1 = sqrt(smoller_riemann_B(jumps) / smoller_riemann_A(jumps))
    lhs = h_1(-log(smoller_riemann_B(jumps)), gas)
    rhs = 2 / (gas.γ - 1) * (1 + c1)
    return lhs < smoller_riemann_C(jumps) < rhs
end

"""
    solve_for_jump_parameters(qL, qR, gas)

Solve for the jump parameters ``(x_1, x_2, x_3)`` as outlined in Smoller.
`qL` and `qR` are already projected into the interface-normal co-ordinates.
"""
function solve_for_jump_parameters(qL, qR, gas)
    Q_jump = JumpRatios(qL, qR, gas)
    x0 = @SVector ones(eltype(qL), 3)
    G(x, Q) =
        let gas = gas
            # 18.62 stuff - A = 0
            v1 = f_1(x[1], gas) * exp(x[2]) * f_3(x[3], gas) - smoller_riemann_A(Q)
            # 18.61: stuff - B = 0 
            v2 = exp(x[3] - x[1]) - smoller_riemann_B(Q)
            # 18.63: stuff - C = 0
            v3 =
                h_1(x[1], gas) +
                sqrt(smoller_riemann_B(Q) / smoller_riemann_A(Q)) *
                h_1(x[1] + log(smoller_riemann_B(Q)), gas) - smoller_riemann_C(Q)
            return SVector(v1, v2, v3)
        end
    nonlinear_system = NonlinearProblem(G, x0, Q_jump)
    solution = solve(nonlinear_system, SimpleNewtonRaphson())
    @assert SciMLBase.successful_retcode(solution.retcode)
    return solution.u
end

function to_primitive_drop_shear(q, gas)
    return (
        ρ = first(q),
        P = dimensionless_pressure(q, gas),
        v = first(select_middle(q)) / first(q),
    )
end

function Euler2D.dimensionless_speed_of_sound(u_prim::NamedTuple{(:ρ, :P, :v)}, gas)
    return sqrt(gas.γ * u_prim.P / u_prim.ρ)
end

function to_conserved(ρ, P, v, v_shear, gas)
    v_all = pushfirst(v_shear, v)
    ρv_all = ρ * v_all
    ρE = P / (gas.γ - 1) + (ρv_all ⋅ v_all) / 2
    return SVector(ρ, ρv_all..., ρE)
end

function to_conserved(u_prim::NamedTuple{(:ρ, :P, :v)}, v_shear, gas)
    return to_conserved(u_prim.ρ, u_prim.P, u_prim.v, v_shear, gas)
end

"""
    shock_speed_via_RH(uL, uR, gas)

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
    rarefaction_1(ray, prim_L, cL, gas)

Find rarefaction structure for states connected to state `L` by a 1-rarefaction. 
"""
function rarefaction_1(ray, prim_L, cL, gas)
    u = ((gas.γ - 1.0) * first(prim_L.v) + 2 * (cL + ray)) / (gas.γ + 1.0)
    ρ = ((prim_L.ρ^gas.γ * (u - ray)^2) / (gas.γ * prim_L.P))^(1 / (gas.γ - 1))
    P = prim_L.P * (ρ / prim_L.ρ)^gas.γ
    return (ρ = ρ, P = P, v = u)
end

"""
    rarefaction_3(ray, prim_R, cR, gas)

Find rarefaction structure for states connected to state `R` by a 3-rarefaction. 
"""
function rarefaction_3(ray, prim_R, cR, gas)
    u = ((gas.γ - 1.0) * first(prim_R.v) - 2 * (cR - ray)) / (gas.γ + 1.0)
    ρ = ((prim_R.ρ^gas.γ * (ray - u)^2) / (gas.γ * prim_R.P))^(1 / (gas.γ - 1))
    P = prim_R.P * (ρ / prim_R.ρ)^gas.γ
    return (ρ = ρ, P = P, v = u)
end

"""
    riemann_states_and_speeds(qL, qR, gas)

Find the information necessary to solve the Riemann problem between `qL` and `qR`.

1. Solve for the jump parameters `[x1, x2, x3]` ([`solve_for_jump_parameters`](@ref)).
2. Strip the shear velocity from `qL` and `qR`; The shear velocity is constant across the contact wave.
3. Apply the transformations to the primitive variables to find the intermediate states.
4. Compute wave speeds and states.

Return a big tuple of relevant values. (shameless rip from the CLAWPACK implementation)
"""
function riemann_states_and_speeds(qL, qR, gas)
    x1, x2, x3 = solve_for_jump_parameters(qL, qR, gas)
    one_shock = x1 < 0
    three_shock = x3 < 0

    # switch to ρ, P, v
    qL_prim = to_primitive_drop_shear(qL, gas)
    qR_prim = to_primitive_drop_shear(qR, gas)
    # apply the transformation functions to find the states between qL and qR
    # that are separated by a contact wave
    qL_prim_★ = T_1(qL_prim, x1, gas)
    qR_prim_★ = T_2(qL_prim_★, x2)
    @assert isapprox(qL_prim_★.v, qR_prim_★.v; atol = 1.0e-8) # assert velocity constant across contact wave!
    # contact discontinuitity propagates at this speed
    contact_speed = qL_prim_★.v
    @assert isapprox(qL_prim_★.P, qR_prim_★.P; atol = 1.0e-8) # assert pressure constant across contact wave!
    # qR_prim_appx = T_3(qR_prim_★..., x3, gas)

    cL = dimensionless_speed_of_sound(qL, gas)
    cR = dimensionless_speed_of_sound(qR, gas)
    cL_★ = dimensionless_speed_of_sound(qL_prim_★, gas)
    cR_★ = dimensionless_speed_of_sound(qR_prim_★, gas)

    speeds_1 = if one_shock
        # solve just the first component of RH
        ρv_L = qL_prim_★.ρ * qL_prim_★.v
        s = (ρv_L - first(select_middle(qL))) / (qL_prim_★.ρ - first(qL))
        (s, s)
    else
        (first(qL_prim.v) - cL, contact_speed - cL_★)
    end

    speeds_3 = if three_shock
        ρv_R = qR_prim_★.ρ * qR_prim_★.v
        s = (first(select_middle(qR)) - ρv_R) / (first(qR) - qR_prim_★.ρ)
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

function solve_riemann_problem(ray, qL, qR, gas)
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

    shear_v_L = popfirst(dimensionless_velocity(qL))
    shear_v_R = popfirst(dimensionless_velocity(qR))
    if ray < contact_speed
        # 1-family
        if has_one_shock
            if ray < sL_1
                return qL
            end
            state = to_conserved(qL_prim_★, shear_v_L, gas)
            return state
        else
            if ray < sL_1
                return qL
            elseif ray < sL_2
                raref_state = rarefaction_1(
                    ray,
                    qL_prim,
                    dimensionless_speed_of_sound(qL_prim, gas),
                    gas,
                )
                state = to_conserved(raref_state, shear_v_L, gas)
                return state
            else
                state = to_conserved(qL_prim_★, shear_v_L, gas)
                return state
            end
        end
    else
        # 3-family
        if has_three_shock
            if ray > sR_2
                return qR
            end
            state = to_conserved(qR_prim_★, shear_v_R, gas)
            return state
        else
            if ray < sR_1
                state = to_conserved(qR_prim_★, shear_v_R, gas)
                return state
            elseif ray < sR_2
                raref_state = rarefaction_3(
                    ray,
                    qR_prim,
                    dimensionless_speed_of_sound(qR_prim, gas),
                    gas,
                )
                state = to_conserved(raref_state, shear_v_R, gas)
                return state
            else
                return qR
            end
        end
    end
end

end

function u_riemann(uL, uR, n, gas)
    qL = change_space_coordinates(uL, n)
    qR = change_space_coordinates(uR, n)
    q_riemann = ExactRiemannSolver.solve_riemann_problem(0, qL, qR, gas)
    u_riemann = change_space_coordinates(q_riemann, n, SVector(1.0, 0.0))
    return u_riemann
end

"""
  ϕ_exact(uL, uR, n, gas)

Compute the value of ``F(u)\\cdot n`` on a boundary with unit normal ``n`` using an exact Riemann solver. 

"""
function ϕ_exact(uL, uR, n, gas)
    return F_euler(u_riemann(uL, uR, n, gas), gas) * n
end
