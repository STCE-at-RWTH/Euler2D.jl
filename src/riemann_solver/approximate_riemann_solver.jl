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

function roe_parameter_vector(u::SVector{N,T}, gas::CaloricallyPerfectGas) where {N,T}
    ρH = dimensionless_total_enthalpy_density(u, gas)
    w = setindex(u, ρH, N) / sqrt(u[1])
    return w
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
    # @assert length(λ_L) == length(λ_R) == length(λ_roe)
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
function ϕ_hll(uL, uR, dim::Int, gas::CaloricallyPerfectGas)
    fL = select_space_dim(F_euler(uL, gas), dim)
    fR = select_space_dim(F_euler(uR, gas), dim)
    sL, sR = interface_signal_speeds(uL, uR, dim, gas)
    return ϕ_hll(uL, uR, @view(fL[:, dim]), @view(fR[:, dim]), sL, sR)
end

"""
    ϕ_hll(uL, uR, n, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `n` : Direction normal to the boundary, points towards uR
"""
function ϕ_hll(uL, uR, n, gas)
    qL = project_state_to_normal(uL, n)
    qR = project_state_to_normal(uR, n)
    fL = F_euler(qL, gas)[:, 1]
    fR = F_euler(qR, gas)[:, 1]
    sL, sR = interface_signal_speeds(qL, qR, 1, gas)
    return ϕ_hll(qL, qR, fL, fR, sL, sR)
end

"""
    ϕ_hll(uL, u̇L, uR, u̇R, dim, gas)

Compute the Jacobian-vector product of `ϕ_hll` given seeds `u̇L` and `u̇R`.

`n` may either be an integer, for which `e1, e2, ` or `e3` will be used as the normal vector pointing towards `uR`, or a unit normal vector.
"""
function ϕ_hll_jvp(uL, u̇L, uR, u̇R, n, gas::CaloricallyPerfectGas)
    u_arg = vcat(uL, uR)
    # TODO how to seed values into ForwardDiff? We shouldn't have to create and multiply a matrix here.
    #   Although, multiplying a 4×8 matrix by an 8×1 vector shouldn't be too bad
    J = jacobian(fdiff_backend, u_arg, Constant(n), Constant(gas)) do u, n, gas
        # TODO remove svector requirement here.
        v1, v2 = split_svector(u)
        return ϕ_hll(v1, v2, n, gas)
    end
    return J * vcat(u̇L, u̇R)
end

"""
    ϕ_hll(uL, uR, fL, fR, dim, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `fL`, `fR`: value of the flux function on either side of the boundary, if known.
- `dim`: Dimension in which to calculate the signal speeds.
"""
function ϕ_hll(uL, uR, fL, fR, dim::Int, gas::CaloricallyPerfectGas)
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
