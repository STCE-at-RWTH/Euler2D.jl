# do we need the multiple eigenvalues in the middle? I do not know...
"""
    eigenvalues_∇F_euler(u, dims, gas)

Computes the eigenvalues of the Jacobian of the Euler flux function in dimension `dim`.
"""
function eigenvalues_∇F_euler(u, dim, gas::CaloricallyPerfectGas)
    ρv = select_middle(u)
    v = ustrip.(ρv / u[1])
    a = ustrip(speed_of_sound(u[1], ρv, u[end], gas))
    return vcat_ρ_ρv_ρE_preserve_static(
        v[dim] - a,
        SVector(ntuple(Returns(v[dim]), length(v))),
        v[dim] + a,
    )
end

function eigenvalues_∇F_euler(
    u::ConservedProps{N,T,Q1,Q2,Q3},
    dim,
    gas::CaloricallyPerfectGas,
) where {N,T,Q1,Q2,Q3}
    v = ustrip.(ShockwaveProperties._units_v, velocity(u))
    a = ustrip(ShockwaveProperties._units_v, speed_of_sound(u, gas))
    return vcat_ρ_ρv_ρE_preserve_static(
        v[dim] - a,
        SVector(ntuple(Returns(v[dim]), N)),
        v[dim] + a,
    )
end

"""
    roe_parameter_vector(u, gas)

Switch from conservation properties `u` to ``[√ρ, √ρ ⋅ v..., √ρ ⋅ H ]``.
We compute enthalphy density `ρH` as the sum of internal energy density and pressure.

See Equations 10 and 11 in Roe.
"""
function roe_parameter_vector(u::ConservedProps, gas::CaloricallyPerfectGas)
    ρH = ustrip(u"J/m^3", total_enthalpy_density(u, gas))
    return SVector{n_space_dims(u) + 2}(
        ustrip(ShockwaveProperties._units_ρ, density(u)),
        ustrip.(ShockwaveProperties._units_ρv, momentum_density(u))...,
        ρH,
    ) / sqrt(ustrip(ShockwaveProperties._units_ρ, density(u)))
end

function roe_parameter_vector(u, gas::CaloricallyPerfectGas)
    ρv = select_middle(u)
    ρH = ustrip(total_enthalpy_density(u[1], ρv, u[end], gas))
    return vcat_ρ_ρv_ρE_preserve_static(u[1], ρv, ρH) / sqrt(u[1])
end

"""
    roe_matrix_eigenvalues(uL, uR, dims, gas)

Find the eigenvalues of the Roe matrix at the boundary determined by ``uL`` and ``uR``. 
"""
function roe_matrix_eigenvalues(uL, uR, dim, gas::CaloricallyPerfectGas)
    wL = roe_parameter_vector(uL, gas)
    wR = roe_parameter_vector(uR, gas)
    w̄ = (wL + wR) / 2
    v̄ = select_middle(w̄) / w̄[1]
    H̄ = w̄[end] / w̄[1]
    a = sqrt((gas.γ - 1) * (H̄ - (v̄ ⋅ v̄) / 2))
    # FIXME how to avoid a huge performance hit here?
    return vcat_ρ_ρv_ρE_preserve_static(
        v̄[dim] - a,
        SVector(ntuple(Returns(v̄[dim]), length(v̄))),
        v̄[dim] + a,
    )
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
    fL = select_space_dim(F_euler(uL, gas), dim)
    fR = select_space_dim(F_euler(uR, gas), dim)
    sL, sR = interface_signal_speeds(uL, uR, dim, gas)
    return ϕ_hll(uL, uR, @view(fL[:, dim]), @view(fR[:, dim]), sL, sR)
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
    return ϕ_hll(uL, uR, select_space_dim(fL, dim), select_space_dim(fR, dim), sL, sR)
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