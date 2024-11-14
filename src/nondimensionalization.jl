struct EulerEqnsScaling{T,Q1<:Length{T},Q2<:Density{T},Q3<:Velocity{T}}
    x_0::Q1
    ρ_0::Q2
    v_0::Q3
end

length_scale(c::EulerEqnsScaling) = c.x_0
density_scale(c::EulerEqnsScaling) = c.ρ_0
velocity_scale(c::EulerEqnsScaling) = c.v_0
time_scale(c::EulerEqnsScaling) = length_scale(c) / velocity_scale(c)
pressure_scale(c::EulerEqnsScaling) = length_scale(c) * velocity_scale(c)^2
energy_density_scale(c::EulerEqnsScaling) = pressure_scale(c) # wow the magic of inviscid flow
body_force_scale(c::EulerEqnsScaling) = energy_density_scale(c) / length_scale(c)

function nondimensionalize(u::ConservedProps{N,T}, s) where {N,T}
    return SVector{N + 2,T}(
        u.ρ / density_scale(s),
        (u.ρv ./ (density_scale(s) * velocity_scale(s)))...,
        u.ρE / energy_density_scale(s),
    )
end

function redimensionalize(u_star::SVector{N,T}, s) where {N,T}
    ρ = u_star[1] * density_scale(s)
    ρv = select_middle(u_star) * density_scale(s) * velocity_scale(s)
    ρE = u_star[N] * energy_density_scale(s)
    return ConservedProps(ρ, ρv, ρE)
end

function _pressure(u_star::SVector{N, T}, gas::CaloricallyPerfectGas)
    return (gas.γ - 1)(u_star[N] - ρv_star ⋅ ρv_star / (2 * u_star[1]))
end

"""
    dimensionless_speed_of_sound(u_star, gas)

Compute the non-dimesionalized speed of sound `a_star` from the nondimesionalized state variable `u``
"""
function dimensionless_speed_of_sound(u_star::SVector{N,T}, gas::CaloricallyPerfectGas)
    P_star = _pressure(u_star, gas)
    return sqrt(gas.γ * (P_star / u_star[1]))
end

"""
    dimensionless_total_enthalpy_density(u_star, gas)

Compute the dimensionless total enthalpy density `ρH_star` from the nondimesionalized state variable `u``
"""
function dimensionless_total_enthalpy_density(u_star::SVector{N, T}, gas::CaloricallyPerfectGas)
    return u_star[N] + _pressure(u_star, gas)
end