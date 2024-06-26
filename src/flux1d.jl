## 1D Riemann problem for the euler equations

using LinearAlgebra
using ShockwaveProperties

"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one column for each space dimension.
"""
function F_euler(u, gas::CaloricallyPerfectGas)
    ρv = @view u[2:end-1]
    v = ρv / u[1]
    P = ustrip(pressure(u[1], ρv, u[end], gas))
    return vcat(ρv', ρv * v' + I * P, (v * (u[end] + P))')
end

function F_euler(u::ConservedProps, gas::CaloricallyPerfectGas)
    P = pressure(u, gas)
    v = velocity(u)
    return vcat(
        momentum_density(u)',
        momentum_density(u) * v' + I * P,
        (v * (total_internal_energy_density(u) + P))',
    )
end

"""
    F_n(u, n, gas)

Computes the flux normal to a given unit vector ``n̂``. Useful for verifying the Rankine-Hugoniot conditions.
"""
F_n(u, n̂, gas::CaloricallyPerfectGas) = F_euler(u, gas) * n̂

# do we need the multiple eigenvalues in the middle? I do not know...
"""
    eigenvalues_∇F(u, dims, gas)

Computes the eigenvalues of the Jacobian of the Euler flux function in each of `dims`, 
which may be a vector/slice of indices or single index.
"""
function eigenvalues_∇F_euler(u, dims, gas::CaloricallyPerfectGas)
    v = @view u[2:end-1]
    a = ustrip(speed_of_sound(u[1], v, u[end], gas))
    out = reduce(vcat, ((v[dims] / u[1])' for i ∈ 1:length(u)))
    @. out[1, :] -= a
    @. out[end, :] += a
    return out
end

"""
    roe_parameter_vector(u, gas)

Switch from conservation properties `u` to ``[√ρ, √ρ ⋅ v..., √ρ ⋅ H ]``.
We compute enthalphy density `ρH` as the sum of internal energy density and pressure.

See Equations 10 and 11 in Roe.
"""
function roe_parameter_vector(u, gas::CaloricallyPerfectGas)
    rhoH = ustrip(total_enthalpy_density(u[1], u[2:end-1], u[end], gas))
    return vcat(u[1], u[2:end-1], rhoH) ./ sqrt(u[1])
end

"""
    roe_matrix_eigenvalues(uL, uR, dims, gas)

Find the eigenvalues of the Roe matrix at the boundary determined by ``uL`` and ``uR``. 
"""
function roe_matrix_eigenvalues(uL, uR, dims, gas::CaloricallyPerfectGas)
    wL = roe_parameter_vector(uL, gas)
    wR = roe_parameter_vector(uR, gas)
    # take arithmetic mean of the left and right states
    w̄ = (wL + wR) / 2
    v̄ = w̄[2:end-1] / w̄[1]
    H̄ = w̄[end] / w̄[1]
    a = sqrt((gas.γ - 1) * (H̄ - (v̄ ⋅ v̄) / 2))
    out = reduce(vcat, (v̄[dims]' for i ∈ eachindex(w̄)))
    @. out[1, :] -= a
    @. out[end, :] += a
    return out
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
    fL = F_euler(uL, gas)[:, dim]
    fR = F_euler(uR, gas)[:, dim]
    sL, sR = interface_signal_speeds(uL, uR, dim, gas)
    return ϕ_hll(uL, uR, fL, fR, sL, sR)
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
    return ϕ_hll(uL, uR, @view(fL[:, dim]), @view(fR[:, dim]), sL, sR)
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
