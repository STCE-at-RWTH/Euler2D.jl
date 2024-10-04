
# these things have the same dimension... weird
MassTransport = MomentumDensity
MomentumTransport = Pressure
@derived_dimension TotalEnergyTransport 𝐌 * 𝐓^-3 true

const _units_ρ_transport = ShockwaveProperties._units_ρv
const _units_ρv_transport = ShockwaveProperties._units_P
const _units_ρE_transport = u"kg/s^3"

# then we can select dimensions easily

struct ConservedPropsTransport{
    N,
    T,
    U1<:MomentumDensity{T},
    U2<:Pressure{T},
    U3<:TotalEnergyTransport{T},
}
    F_ρ::SVector{N,U1}
    F_ρv::SMatrix{N,N,U2}
    F_ρE::SVector{N,U3}
end

n_space_dims(F_e::ConservedPropsTransport{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N

function transport_to_matrix(s::ConservedPropsTransport)
    return vcat(s.F_ρ', s.F_ρv, s.F_ρE')
end

"""
    F_euler(u::ConservedProps, gas::CaloricallyPerfectGas)

Computes the value of the flux function F(u) for the Euler equations.
Returns a `ConservedPropsTransport` that maintains the unitful-ness of the quantity.
"""
function F_euler(
    u::ConservedProps{N,T,U1,U2,U3},
    gas::CaloricallyPerfectGas,
) where {N,T,U1,U2,U3}
    ρv = momentum_density(u)
    P = pressure(u, gas)
    v = velocity(u)
    vρv = SMatrix{N,N}(momentum_density(u) * v' + I * P)
    vρE = SVector{N}(v * (total_internal_energy_density(u) + P))
    return ConservedPropsTransport(ρv, vρv, vρE)
end

# TODO make this return a "normed" ConservedPropsTransport struct?

function F_euler_normal(F_e::ConservedPropsTransport{1,T,U1,U2,U3}, n̂) where {T,U1,U2,U3}
    return @SVector [
        ustrip(_units_ρ_transport, F_e.F_ρ[1] * n̂[1]),
        ustrip(_units_ρv_transport, F_e.F_ρv[1] * n̂[1]),
        ustrip(_units_ρE_transport, F_e.F_ρE * n̂[1]),
    ]
end

function F_euler_normal(F_e::ConservedPropsTransport, n̂)
    F_ρ = ustrip(_units_ρ_transport, F_e.F_ρ ⋅ n̂)
    F_ρv = ustrip.(_units_ρv_transport, F_e.F_ρv * n̂)
    F_ρE = ustrip.(_units_ρE_transport, F_e.F_ρE ⋅ n̂)
    return SVector{n_space_dims(F_e) + 2}(F_ρ, F_ρv..., F_ρE)
end

"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one column for each space dimension.

This implementation will strip out units, and convert down to metric base units in the process. 
"""
function F_euler(u::StaticVector{S,T}, gas::CaloricallyPerfectGas) where {S,T}
    ρv = select_middle(u)
    v = SVector{S - 2,T}(ρv / u[1])
    P = ustrip(ShockwaveProperties._units_P, pressure(u[1], ρv, u[end], gas))
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
    F_euler(u, P)

Compute the Euler flux with a provided value for pressure.
Intended for use when a gas has a fixed, "theoretical" value for `γ`
    that allows easy computation of `P`.
"""
function F_euler(u, P)
    ρv = @view(u[2:end-1])
    ρv_flux = (ρv * ρv') / u[1] + I*P
    return vcat(ρv', ρv_flux, (ρv *(u[end] + P)/u[1])')
end

F_euler_normal(F_e, n̂) = F_e * n̂

"""
    F_n(u, n, gas)

Computes the flux normal to a given unit vector ``n̂``. Useful for verifying the Rankine-Hugoniot conditions.
"""
F_n(u, n̂, gas::CaloricallyPerfectGas) = F_euler_normal(F_euler(u, gas), n̂)

# TODO this allocates, is type unstable
function select_space_dim(
    F_e::ConservedPropsTransport{N,T,U1,U2,U3},
    dim,
) where {N,T,U1,U2,U3}
    idxs = SVector(ntuple(i -> i, N))
    return vcat_ρ_ρv_ρE_preserve_static(
        ustrip(_units_ρ_transport, F_e.F_ρ[dim]),
        ustrip.(_units_ρv_transport, F_e.F_ρv[idxs, dim]),
        ustrip(_units_ρE_transport, F_e.F_ρE[dim]),
    )
end

select_space_dim(F_e, dim) = F_e[:, dim]