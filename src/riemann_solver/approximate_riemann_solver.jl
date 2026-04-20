"""
    shock_and_rarefaction_speeds(uL, uR, dim, gas)

Estimate the signal speeds in the 1- and 3-wave families at the interface between `uL` and `uR` in dimension `dim`.

Computed according to Einfeldt's approximations, listed as **2.24** in *Vides, Nkonga & Audit*. 
These compare the eigenvalues of the Jacobian of the flux function to the 
eigenvalues of the Roe matrix and pick the "faster" speed.
"""
function shock_and_rarefaction_speeds(uL, uR, dim, gas::CaloricallyPerfectGas)
    λ_roe = roe_average_eigenvalues(uL, uR, dim, gas)
    λ_L = eigenvalues_∇F_euler(uL, dim, gas)
    λ_R = eigenvalues_∇F_euler(uR, dim, gas)
    # @assert length(λ_L) == length(λ_R) == length(λ_roe)
    # 2.24 from Vides, et al.
    s_L = min((min(λ...) for λ in zip(λ_L, λ_roe))...)
    s_R = max((max(λ...) for λ in zip(λ_roe, λ_R))...)
    return s_L, s_R
end

"""
    contact_wave_speed(uL, uR, dim, gas)

Estimate the contact wave speed between two states `uL` and `uR` in dimension `dim`
according to Einfeldt. See equations 4 through 8 in [Fleischmann et al](https://doi.org/10.1016/j.compfluid.2019.04.004).
"""
function contact_wave_speed(uL, uR, dim::Int, gas)
    ρL = first(uL)
    ρR = first(uR)
    PL = dimensionless_pressure(uL, gas)
    PR = dimensionless_pressure(uR, gas)
    vL = dimensionless_velocity(uL)[dim]
    vR = dimensionless_velocity(uR)[dim]
    sL, sR = shock_and_rarefaction_speeds(uL, uR, dim, gas)
    s★ = (PR - PL + ρL * vL * (sL - vL) - ρR * vR * (sR - vR)) / (ρL * (sL - vL) - ρR * (sR - vR))
    return s★
end

"""
    ϕ_hll(uL, uR, dim, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `dim` : Direction to calculate F_hll
"""
function ϕ_hll(uL, uR, dim::Int, gas::CaloricallyPerfectGas)
    fL = F_euler(uL, gas)[:, dim]
    fR = F_euler(uR, gas)[:, dim]
    sL, sR = shock_and_rarefaction_speeds(uL, uR, dim, gas)
    return ϕ_hll(uL, uR, fL, fR, sL, sR)
end

"""
    ϕ_hll(uL, uR, n, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `n` : Direction normal to the boundary, assumed that it points towards `uR`
"""
function ϕ_hll(uL, uR, n, gas)
    qL = change_space_coordinates(uL, n)
    qR = change_space_coordinates(uR, n)
    fL = F_euler(qL, gas)[:, 1]
    fR = F_euler(qR, gas)[:, 1]
    sL, sR = shock_and_rarefaction_speeds(qL, qR, 1, gas)
    return ϕ_hll(qL, qR, fL, fR, sL, sR)
end

"""
    ϕ_hll_and_jvp(uL, u̇L, uR, u̇R, dim, gas)

Compute the Jacobian-vector product of `ϕ_hll` given seeds `u̇L` and `u̇R`.

`n` may either be an integer, for which `e1, e2, ` or `e3` will be used as the normal vector pointing towards `uR`, or a unit normal vector.
(I think).
"""
function ϕ_hll_and_jvp(uL, u̇L, uR, u̇R, n, gas::CaloricallyPerfectGas)
    u_arg = vcat(uL, uR)
    u̇_arg = vcat(u̇L, u̇R)
    res, jvp = DifferentiationInterface.value_and_pushforward(
        fdiff_backend,
        u_arg,
        ntuple(i -> u̇_arg[:, i], size(u̇_arg)[2]),
        Constant(n),
        Constant(gas),
    ) do u_arg, n, gas
        v1, v2 = split_svector(u_arg)
        return ϕ_hll(v1, v2, n, gas)
    end
    return res, jvp
end

"""
    ϕ_hll(uL, uR, fL, fR, dim, gas)

Compute the HLL numerical flux across the L-R boundary.
- `uL`, `uR`: States on either side of the boundary
- `fL`, `fR`: value of the flux function on either side of the boundary, if known.
- `dim`: Dimension in which to calculate the signal speeds.
"""
function ϕ_hll(uL, uR, fL, fR, dim::Int, gas::CaloricallyPerfectGas)
    sL, sR = shock_and_rarefaction_speeds(uL, uR, dim, gas)
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

function u_hllc(u::SVector{N}, s, s★, dim, gas) where {N}
    v = dimensionless_velocity(u)[dim]
    P = dimensionless_pressure(u, gas)
    ρ★ = first(u)
    ρv★ = setindex(select_middle(u), ρ★ * s★, dim)
    ρE★ = last(u) + (s★ - v) * (ρ★ * s★ + P / (s - v))
    u★ = (s - v) / (s - s★) * vcat_state_components(ρ★, ρv★, ρE★)
    return u★
end

function ϕ_hllc(uL, uR, dim::Int, gas)
    sL, sR = shock_and_rarefaction_speeds(uL, uR, dim, gas)
    sC = contact_wave_speed(uL, uR, dim, gas)
    fL = F_euler(uL, gas)[:, dim]
    fR = F_euler(uR, gas)[:, dim]
    s_minus = min(sL, 0)
    s_plus = max(sR, 0)
    uL★ = u_hllc(uL, sL, sC, dim, gas)
    uR★ = u_hllc(uR, sR, sC, dim, gas)
    f_hllc_l = fL + s_minus * (uL★ - uL)
    f_hllc_r = fR + s_plus * (uR★ - uR)
    # equation 4 in Fleischmann
    return (1 + sign(sC)) / 2 * f_hllc_l + (1 - sign(sC)) / 2 * f_hllc_r
end

function ϕ_hllc_and_jvp(uL, udotL::SMatrix{4, NS}, uR, udotR::SMatrix{4, NS}, n, gas) where {NS}
    u_arg = vcat(uL, uR)
    u̇_arg = vcat(udotL, udotR)
    res, jvp = DifferentiationInterface.value_and_pushforward(
        fdiff_backend,
        u_arg,
        ntuple(i -> u̇_arg[:, i], NS),
        Constant(n),
        Constant(gas),
    ) do u_arg, n, gas
        v1, v2 = split_svector(u_arg)
        return ϕ_hllc(v1, v2, n, gas)
    end
    return res, jvp
end
