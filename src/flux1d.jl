using Unitful: Temperature, Pressure, Density, Velocity, @derived_dimension
using Unitful: 𝐋, 𝐓, 𝐌, 𝚯, 𝐍

"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one column for each space dimension.

This will strip out units, and convert down to metric base units in the process. 
"""
function F_euler(u, gas::CaloricallyPerfectGas)
    ρv = SVector{length(u) - 2}(@view(u[2:end-1])...)
    v = SVector{length(ρv)}(ρv / u[1])
    P = ustrip(u"Pa", pressure(u[1], ρv, u[end], gas))
    ρv_flux = ρv * v' + I * P
    return vcat(ρv', ρv_flux, (v * (u[end] + P))')
end

function F_euler(u::ConservedProps, gas::CaloricallyPerfectGas)
    P = ustrip(u"Pa", pressure(u, gas))
    v = ustrip.(u"m/s", velocity(u))
    momentum_flux = ustrip.(ShockwaveProperties._units_ρv, momentum_density(u)) * v' + I * P
    return vcat(
        ustrip.(ShockwaveProperties._units_ρv, momentum_density(u))',
        momentum_flux,
        (
            v *
            (ustrip(ShockwaveProperties._units_ρE, total_internal_energy_density(u)) + P)
        )',
    )
end

"""
    apply_units_F(F; units)

Re-apply SI base units to the columns of the flux array.
"""
function apply_units_F(
    F_unitless;
    units = (1.0u"kg*m^-2*s^-1", 1.0u"kg*m^-1*s^2", 1.0u"kg*m^-1*s^2", 1.0u"kg*s^-3"),
)
    return stack(eachcol(F_unitless)) do Fi
        Fi .* units
    end
end

"""
    F_n(u, n, gas)

Computes the flux normal to a given unit vector ``n̂``. Useful for verifying the Rankine-Hugoniot conditions.
"""
F_n(u, n̂, gas::CaloricallyPerfectGas) = F_euler(u, gas) * n̂

# do we need the multiple eigenvalues in the middle? I do not know...
"""
    eigenvalues_∇F(u, dims, gas)

Computes the eigenvalues of the Jacobian of the Euler flux function in dimension `dim`.
"""
function eigenvalues_∇F_euler(u, dim, gas::CaloricallyPerfectGas)
    v = @view u[2:end-1]
    a = ustrip(speed_of_sound(u[1], v, u[end], gas))
    return SVector(v[dim] - a, ntuple(Returns(v[dim]), length(v))..., v[dim] + a)
end

function eigenvalues_∇F_euler(
    u::ConservedProps{N,T,Q1,Q2,Q3},
    dim,
    gas::CaloricallyPerfectGas,
) where {N,T,Q1,Q2,Q3}
    v = ustrip.(u"m/s", velocity(u))
    a = ustrip(u"m/s", speed_of_sound(u, gas))
    return SVector(v[dim] - a, ntuple(Returns(v[dim])..., length(v))..., v[dim] + a)
end

"""
    roe_parameter_vector(u, gas)

Switch from conservation properties `u` to ``[√ρ, √ρ ⋅ v..., √ρ ⋅ H ]``.
We compute enthalphy density `ρH` as the sum of internal energy density and pressure.

See Equations 10 and 11 in Roe.
"""
function roe_parameter_vector(u, gas::CaloricallyPerfectGas)
    rhoH = ustrip(total_enthalpy_density(u[1], @view(u[2:end-1]), u[end], gas))
    return SVector(u[1], @view(u[2:end-1])..., rhoH) ./ sqrt(u[1])
end

@derived_dimension RoeDensity 𝐌^(1 / 2) * 𝐋^(-3 / 2)
@derived_dimension RoeMomentum 𝐌^(1 / 2) * 𝐋^(-1 / 2) * 𝐓^-1
@derived_dimension RoeEnergy 𝐌^(1 / 2) * 𝐋^(1 / 2) * 𝐓^-2

"""
    RoeProps{N, T, U1, U2, U3}
"""
struct RoeProps{N,DTYPE,U1<:RoeDensity{DTYPE},U2<:RoeMomentum{DTYPE},U3<:RoeEnergy{DTYPE}}
    ρ::U1
    ρv::SVector{N,U2}
    ρE::U3
end

function roe_parameter_vector(
    u::ConservedProps{N,T,Q1,Q2,Q3},
    gas::CaloricallyPerfectGas,
) where {N,T,Q1,Q2,Q3}
    ρH = ustrip(ShockwaveProperties._units_ρE, total_enthalpy_density(u, gas))
    root_rho = sqrt(ustrip(ShockwaveProperties._units_ρ, density(u)))
    return SVector{N + 2}(
        root_rho,
        (ustrip.(ShockwaveProperties._units_ρv, momentum_density(u)) ./ root_rho)...,
        ρH / root_rho,
    )
end

"""
    roe_matrix_eigenvalues(uL, uR, dims, gas)

Find the eigenvalues of the Roe matrix at the boundary determined by ``uL`` and ``uR``. 
"""
function roe_matrix_eigenvalues(uL, uR, dim, gas::CaloricallyPerfectGas)
    wL = roe_parameter_vector(uL, gas)
    wR = roe_parameter_vector(uR, gas)
    # take arithmetic mean of the left and right states
    w̄ = (wL + wR) / 2
    v̄ = SVector(@view(w̄[2:end-1])...) / w̄[1]
    H̄ = w̄[end] / w̄[1]
    a = sqrt((gas.γ - 1) * (H̄ - (v̄ ⋅ v̄) / 2))
    return SVector(v̄[dim] - a, ntuple(Returns(v̄[dim]), length(uL) - 2)..., v̄[dim] + a)
end

"""
    interface_signal_speeds(uL, uR, dim, gas)

Compute the left and right signal speeds at the interface between `uL` and `uR` in dimension `dim`.

Computed according to Einfeldt's approximations, listed as **2.24** in *Vides, Nkonga & Audit*. 
These compare the eigenvalues of the Jacobian of the flux function to the 
eigenvalues of the Roe matrix and pick the "faster" speed.
"""
function interface_signal_speeds(uL, uR, dim, gas::CaloricallyPerfectGas)
    λ_roe = roe_matrix_eigenvalues(uL, uR, dim, gas)
    λ_L = eigenvalues_∇F_euler(uL, dim, gas)
    λ_R = eigenvalues_∇F_euler(uR, dim, gas)
    @assert length(λ_L) == length(λ_R) == length(λ_roe)
    # 2.24 from Vides, et al.
    s_L = min((min(λ...) for λ ∈ zip(λ_L, λ_roe))...)
    s_R = max((max(λ...) for λ ∈ zip(λ_roe, λ_R))...)
    return s_L, s_R
end

"""
    ϕ_hll(uL, uR, dim, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `dim` : Direction to calculate F_hll
"""
function ϕ_hll(uL, uR, dim, gas::CaloricallyPerfectGas)
    fL = F_euler(uL, gas)
    fR = F_euler(uR, gas)
    sL, sR = interface_signal_speeds(uL, uR, dim, gas)
    return ϕ_hll(
        uL,
        uR,
        SVector(@view(fL[:, dim])...),
        SVector(@view(fR[:, dim])...),
        sL,
        sR,
    )
end

"""
    ϕ_hll(uL, uR, fL, fR, dim, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `fL`, `fR`: value of the flux function on either side of the boundary, if known.
- `dim`: Dimension in which to calculate the signal speeds.
"""
function ϕ_hll(uL, uR, fL, fR, dim, gas::CaloricallyPerfectGas)
    sL, sR = interface_signal_speeds(uL, uR, dim, gas)
    return ϕ_hll(
        uL,
        uR,
        SVector(@view(fL[:, dim])...),
        SVector(@view(fR[:, dim])...),
        sL,
        sR,
    )
end

"""
    ϕ_hll(uL, uR, fL, fR, sL, sR, dim, gas)

Compute the HLL numerical flux across the L-R boundary and correct for the supersonic case.
- `uL`, `uR`: States on either side of the boundary
- `fL`, `fR`: value of the flux function on either side of the boundary, if known.
- `sL`, `sR`: left and right signal speeds at the boundary, if known.

_Equation **2.20** from Vides et al._
"""
function ϕ_hll(uL, uR, fL, fR, sL, sR)
    @assert sL < sR
    # we leave this as an if statement to avoid many flops
    if sR < 0
        # flow only into left cell
        return fR
    elseif sL > 0
        # flow only into right cell
        return fL
    else
        # shared flow
        return (sR * fL - sL * fR + sR * sL * (uR - uL)) / (sR - sL)
    end
end
