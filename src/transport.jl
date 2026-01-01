
# these things have the same dimension... weird
# MassTransport = MomentumDensity
# MomentumTransport = Pressure
# @derived_dimension TotalEnergyTransport 𝐌 * 𝐓^-3 true
#
# const _units_ρ_transport = ShockwaveProperties._units_ρv
# const _units_ρv_transport = ShockwaveProperties._units_P
# const _units_ρE_transport = u"kg/s^3"

# then we can select dimensions easily
# struct ConservedPropsTransport{
#     N,
#     T,
#     U1<:MomentumDensity{T},
#     U2<:Pressure{T},
#     U3<:TotalEnergyTransport{T},
#     NSQR,
# }
#     F_ρ::SVector{N,U1}
#     F_ρv::SMatrix{N,N,U2,NSQR}
#     F_ρE::SVector{N,U3}
# end

# n_space_dims(::ConservedPropsTransport{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N

# function transport_to_matrix(s::ConservedPropsTransport)
# return vcat(s.F_ρ', s.F_ρv, s.F_ρE')
# end

# """
#     F_euler(u::ConservedProps, gas::CaloricallyPerfectGas)
#
# Computes the value of the flux function F(u) for the Euler equations.
# Returns a `ConservedPropsTransport` that maintains the unitful-ness of the quantity.
#
# DEPRECATED DO NOT DO THIS PLEASE
# """
# function F_euler(
#     u::ConservedProps{N,T,U1,U2,U3},
#     gas::CaloricallyPerfectGas,
# ) where {N,T,U1,U2,U3}
#     ρv = momentum_density(u)
#     P = pressure(u, gas)
#     v = velocity(u)
#     vρv = SMatrix{N,N}(momentum_density(u) * v' + I * P)
#     vρE = SVector{N}(v * (total_internal_energy_density(u) + P))
#     return ConservedPropsTransport(ρv, vρv, vρE)
# end
#

# function F_euler_normal(F_e::ConservedPropsTransport{1,T,U1,U2,U3}, n̂) where {T,U1,U2,U3}
#     return @SVector [
#         ustrip(_units_ρ_transport, F_e.F_ρ[1] * n̂[1]),
#         ustrip(_units_ρv_transport, F_e.F_ρv[1] * n̂[1]),
#         ustrip(_units_ρE_transport, F_e.F_ρE * n̂[1]),
#     ]
# end

# function F_euler_normal(F_e::ConservedPropsTransport, n̂)
#     F_ρ = ustrip(_units_ρ_transport, F_e.F_ρ ⋅ n̂)
#     F_ρv = ustrip.(_units_ρv_transport, F_e.F_ρv * n̂)
#     F_ρE = ustrip.(_units_ρE_transport, F_e.F_ρE ⋅ n̂)
#     return SVector{n_space_dims(F_e) + 2}(F_ρ, F_ρv..., F_ρE)
# end

"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one column for each space dimension.

This implementation will strip out units, and convert down to metric base units in the process. 
"""
function F_euler(u::SVector{S,T}, gas::CaloricallyPerfectGas) where {S,T}
    ρv = select_middle(u)
    v = SVector{S - 2,T}(ρv / u[1])
    P = dimensionless_pressure(u, gas)
    ρv_flux = ρv * v' + I * P
    return vcat(ρv', ρv_flux, (v * (u[end] + P))')
end

function F_euler(u, gas::CaloricallyPerfectGas)
    ρv = @view(u[2:end-1])
    P = ustrip(ShockwaveProperties._units_P, pressure(u[1], ρv, u[end], gas))
    ρv_flux = (ρv * ρv') / u[1] + I * P
    return vcat(ρv', ρv_flux, (ρv * (u[end] + P) / u[1])')
end

"""
    F_euler_n(u, n, gas)

Computes the flux normal to a given unit vector ``n̂``. Useful for verifying the Rankine-Hugoniot conditions.
"""
F_euler_n(u, n̂, gas::CaloricallyPerfectGas) = F_euler(u, gas) * n̂

# TODO this allocates, is type unstable
# function select_space_dim(
#     F_e::ConservedPropsTransport{N,T,U1,U2,U3},
#     dim,
# ) where {N,T,U1,U2,U3}
#     idxs = SVector(ntuple(i -> i, N))
#     return vcat_state_components(
#         ustrip(_units_ρ_transport, F_e.F_ρ[dim]),
#         ustrip.(_units_ρv_transport, F_e.F_ρv[idxs, dim]),
#         ustrip(_units_ρE_transport, F_e.F_ρE[dim]),
#     )
# end

select_space_dim(F_e, dim) = F_e[:, dim]

# do we need the multiple eigenvalues in the middle? I do not know...
"""
    eigenvalues_∇F_euler(u, dims, gas)

Computes the eigenvalues of the Jacobian of the Euler flux function in dimension `dim`.
"""
function eigenvalues_∇F_euler(u, dim, gas::CaloricallyPerfectGas)
    ρv = select_middle(u)
    v = ρv / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    return SVector(v[dim] - a, ntuple(Returns(v[dim]), length(v))..., v[dim] + a)
end

function eigenvalues_∇F_euler(
    u::ConservedProps{N,T,Q1,Q2,Q3},
    dim,
    gas::CaloricallyPerfectGas,
) where {N,T,Q1,Q2,Q3}
    v = ustrip.(ShockwaveProperties._units_v, velocity(u))
    a = ustrip(ShockwaveProperties._units_v, speed_of_sound(u, gas))
    return SVector(v[dim] - a, ntuple(Returns(v[dim]), N)..., v[dim] + a)
end

"""
  eigenvectors_∇F_euler(u, gas)

Compute the eigenvectors of the Jacobian of the Euler equations flux.
"""
function eigenvectors_∇F_euler(u::SVector{3,T}, gas) where {T}
    v = u[2] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = SVector(1, v - a, H - v * a)
    r2 = SVector(1, v, v * v / 2)
    r3 = SVector(1, v + a, H + v * a)
    return hcat(r1, r2, r3)
end

"""
Compute the eigenvectors of the Jacobian of the x-component of the Euler equations flux.
"""
function eigenvectors_∇F_euler(u::SVector{4,T}, gas) where {T}
    v1 = u[2] / u[1]
    v2 = u[3] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = SVector(1.0, v1 - a, v2, H - a * v1)
    r2 = SVector(1.0, v1, v2, 0.5 * (v1 * v1 + v2 * v2))
    r3 = SVector(0.0, 0.0, 1.0, v2)
    r4 = SVector(1.0, v1 + a, v2, H + a * v1)
    return hcat(r1, r2, r3, r4)
end

"""
  eigenvectors_∇G_euler(u, gas)

Compute the eigenvectors of the Jacobian of the y-component of the Euler equations flux.
"""
function eigenvectors_∇G_euler(u::SVector{4,T}, gas) where {T}
    v1 = u[2] / u[1]
    v2 = u[3] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = SVector(1.0, v1, v2 - a, H - a * v2)
    r2 = SVector(0.0, 1.0, 0.0, v1)
    r3 = SVector(1.0, v1, v2, 0.5 * (v1 * v1 + v2 * v2))
    r4 = SVector(1.0, v1, v2 + a, H + a * v2)
    return hcat(r1, r2, r3, r4)
end
