module ShockSensitivity

using LinearAlgebra

using DifferentiationInterface
using Integrals
using SimpleNonlinearSolve
using StaticArrays

using PlanePolygons

using Euler2D
using Euler2D: BowShockInterpolation, CannyShockSensor
using Euler2D: CaloricallyPerfectGas
using Euler2D: change_space_coordinates
using Euler2D: fdiff_backend

using .CannyShockSensor: hugoniot_equation_relative_err

"""
    F_euler_θ

Changes `u` to the coordinate system given with ``e_1 = n`` or by some basis matrix `B` and then calculates `F_euler`.
"""
function F_euler_ξ1(u, gas)
    return F_euler(u, gas)[:, 1]
end

function F_euler_ξ1(u, new_coordinates, gas)
    return F_euler(change_space_coordinates(u, new_coordinates), gas)[:, 1]
end

function _avg_jacobian_igrand(ϑ, params)
    (uL, uR, gas) = params
    arg = ϑ * uL + (1 - ϑ) * uR
    return jacobian(F_euler_ξ1, fdiff_backend, arg, Constant(gas))
end

function avg_jacobian_F_euler(qL, qR, gas)
    domain = (0.0, 1.0)
    problem = IntegralProblem(_avg_jacobian_igrand, domain, (qL, qR, gas))
    sol = solve(problem, HCubatureJL())
    @assert SciMLBase.successful_retcode(sol)
    return sol.u
end

_F_e1(u, gas) = F_euler(u, gas)[:, 1]

function _∇F_e1(u, gas)
    return jacobian(_F_e1, fdiff_backend, u, Constant(gas))
end

function _F_e1_and_jvp(u, udot, gas)
    (F, (Fdot,)) = value_and_pushforward(_F_e1, fdiff_backend, u, (udot,), Constant(gas))
    return F, Fdot
end

function _change_space_coords_jvp(u, udot, T_or_e1)
    (u_θ, (udot_θ,)) = value_and_pushforward(
        Euler2D.change_space_coordinates,
        fdiff_backend,
        u,
        (udot,),
        Constant(T_or_e1),
    )
    return u_θ, udot_θ
end

_total_dp_U(dU_ds, Γdot, Udot) = dU_ds * Γdot + Udot
_total_dp_Fn(dF, dU_ds, Γdot, Udot) = dF .* _total_dp_U(dU_ds, Γdot, Udot)

"""
    ShockSensitivityStep{N, T, GAS_T}

Functor struct for computing the shock sensitivity time step from Hüser; Section 6, Definition 39.

Fields
---
- `n`: the unit normal vector to the shock for which we wish to compute the sensitivity in standard coordinates. Points towards the "right".
- `U_L`, `U_R`: the (estimated) value of `U` on either side of the shock.
- `Udot_L, `Udot_R`: the (estimated) value of the accumulated tangent of `U` on either side of the shock.
- `dU_Lds`, `dU_Rds`: the (estimated) value of `∇U` along `n` on either side of the shock.
"""
struct ShockSensitivityStep{N, T, GAS_T}
    n::SVector{2, T} # the new choice of +x for this calculation in STANDARD BASIS
    gas::GAS_T
    U_L::SVector{N, T}
    U_R::SVector{N, T}
    Udot_L::SVector{N, T}
    Udot_R::SVector{N, T}
    dU_Lds::SVector{N, T}
    dU_Rds::SVector{N, T}
end

"""
    ShockSensitivityStep(shock_point, shock_normal, pdot, Cr, Δs, alpha, sim, gas)

Construct a `ShockSensitivityStep` functor for a shock located at `shock_point` with right-facing unit normal `shock_normal`.
Does the following:
1. Get `U_L` and `U_R` and their pushforwards from `sim` by looking outside of the shock region `±Cr*Δs^alpha`.
2. Get `U_LL` and `U_RR` by looking `2Δs` further from the shock region.
3. Project all four states (and pushforwards) into a coordinate system with `e1 = n`.
4. Compute `dU_Lds` and `dU_Rds` ahead-of-time.

Arguments
---
- `shock_point`: Point on the shock
- `shock_normal`: Unit normal vector to the shock.
- `pdot`: Direction in which the parameters are perturbed.
- `Cr`: Constant chosen to satisfy the requirements set out for the shock sensitivity step algorithm.
- `Δs`: Step size on the shock normal axis.
- `alpha`: Exponent chosen to satisfy the requirements set out for the shock sensitivity step algorithm.
- `sim`: The simulation in question.
- `gas`: The gas which the simulation was done with. (I think this only requires a field `γ`, so `NamedTuple` should work...)
"""
function ShockSensitivityStep(shock_point, shock_normal, pdot, Cr, Δs, alpha, sim, gas)
    Cr_ds_alpha = Cr * Δs^alpha
    shock_normal_axis = Line(shock_point, shock_normal)
    U_L_xy, Udot_L_xy = Euler2D.u_and_u̇_at(shock_normal_axis(-Cr_ds_alpha)..., sim, 1)
    U_R_xy, Udot_R_xy = Euler2D.u_and_u̇_at(shock_normal_axis(Cr_ds_alpha)..., sim, 1)
    U_LL_xy = Euler2D.u_at(shock_normal_axis(-Cr_ds_alpha - 2 * Δs)..., sim, 1)
    U_RR_xy = Euler2D.u_at(shock_normal_axis(Cr_ds_alpha + 2 * Δs)..., sim, 1)
    U_L, Udot_L = _change_space_coords_jvp(U_L_xy, Udot_L_xy * pdot, shock_normal)
    U_R, Udot_R = _change_space_coords_jvp(U_R_xy, Udot_R_xy * pdot, shock_normal)
    U_LL = Euler2D.change_space_coordinates(U_LL_xy, shock_normal)
    U_RR = Euler2D.change_space_coordinates(U_RR_xy, shock_normal)
    dU_Lds = (U_L - U_LL) / (2 * Δs)
    dU_Rds = (U_RR - U_R) / (2 * Δs)
    return ShockSensitivityStep(shock_normal, gas, U_L, U_R, Udot_L, Udot_R, dU_Lds, dU_Rds)
end

"""
    (data::ShockSensitivityStep)(Γdot)

Compute `ΔΓdot` from `Γdot`.
"""
function (ΔΓdot_data::ShockSensitivityStep{N, T, G})(Γdot) where {N, T, G}
    dU_L_dp = _total_dp_U(ΔΓdot_data.dU_Lds, Γdot, ΔΓdot_data.Udot_L)
    dU_R_dp = _total_dp_U(ΔΓdot_data.dU_Rds, Γdot, ΔΓdot_data.Udot_R)
    F_L, dF_L_dp = _F_e1_and_jvp(ΔΓdot_data.U_L, dU_L_dp, ΔΓdot_data.gas)
    F_R, dF_R_dp = _F_e1_and_jvp(ΔΓdot_data.U_R, dU_R_dp, ΔΓdot_data.gas)
    ΔU = ΔΓdot_data.U_L - ΔΓdot_data.U_R
    ΔdU_dp = dU_L_dp - dU_R_dp
    ΔFn = F_L - F_R
    ΔdFn_dp = dF_L_dp - dF_R_dp
    return @. inv(ΔU) * ΔdFn_dp - inv(ΔU^2) * ΔFn * ΔdU_dp
end

@doc raw"""
    verify_stationary_shock_sensitivity(Γdot_guess, data)

Eventually, we should be able to set ``\Delta \Gamma^{(1)} = 0``, since the shock is stationary.
After some brief algebriac manipulation we end up with 

```math
\Nabla F_{L,k}d_pU_{L,k}-\Nabla F_{R,k}d_pU_{R,k} = (U_{L,k}-U_{R,k})^{-1}\cdot(F_{L,k}-F_{R,k})\cdot(d_pU_{L,k}-d_pU_{R,k})
```
"""
function verify_stationary_shock_sensitivity(Γdot_guess, data)
    dU_L_dp = _total_dp_U(data.dU_Lds, Γdot_guess, data.Udot_L)
    dU_R_dp = _total_dp_U(data.dU_Rds, Γdot_guess, data.Udot_R)
    Fn_L, dFn_L_dp = _F_e1_and_jvp(data.U_L, dU_L_dp, data.gas)
    Fn_R, dFn_R_dp = _F_e1_and_jvp(data.U_R, dU_R_dp, data.gas)
    ΔU = data.U_L - data.U_R
    ΔdU_dp = dU_L_dp - dU_R_dp
    ΔFn = Fn_L - Fn_R
    ΔdFn_dp = dFn_L_dp - dFn_R_dp
    return @. ΔdFn_dp - inv(ΔU) * ΔFn * ΔdU_dp
end

function verify_jump_properties(data)
    hugoniot_error = hugoniot_equation_relative_err(data.U_L, data.U_R, data.gas)
    P_ratio =
        dimensionless_pressure(data.U_R, data.gas) /
        dimensionless_pressure(data.U_L, data.gas)
    P_ratio_ideal = ShockwaveProperties.shock_pressure_ratio(
        dimensionless_mach_number(data.U_L, data.gas),
        SVector(1.0, 0.0),
        data.gas,
    )

    P_ratio_error = abs(1 - P_ratio / P_ratio_ideal)
    vL = dimensionless_velocity(data.U_L)
    vR = dimensionless_velocity(data.U_R)
    velocity_error = abs(1 - last(vL) / last(vR))

    return hugoniot_error, P_ratio_error, velocity_error
end
end
