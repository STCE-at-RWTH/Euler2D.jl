"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one column for each space dimension.

This implementation will strip out units, and convert down to metric base units in the process. 
"""
function F_euler(u::SVector{S,T}, gas::CaloricallyPerfectGas) where {S,T}
    ρv = select_middle(u)
    v = ρv / u[1]
    P = dimensionless_pressure(u, gas)
    ρv_flux = ρv * v' + I * P
    return vcat(ρv', ρv_flux, (v * (u[end] + P))')
end

# we could definitely clean up the allocs here but it's better than it was now
function F_euler(u, gas)
    F = similar(u, (length(u), length(u) - 2))
    ρ = first(u)
    ρv = select_middle(u)
    ρE = last(u)
    P = dimensionless_pressure(u, gas)
    # shout out broadcast fusion
    @. F[begin, :] = ρv
    F[begin+1:end-1, :] .= (ρv .* ρv') ./ ρ
    F[begin+1:end-1, :] += I * P
    c1 = (ρE + P) / ρ
    @. F[end, :] = ρv * c1
    return F
end

function F_euler(u::ConservedProps, gas, scale)
    return F_euler(nondimensionalize(u, scale), gas)
end

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
