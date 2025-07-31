"""
  eigenvectors_∇F_euler(u, gas)

Compute the eigenvectors of the Jacobian of the Euler equations flux.
"""
function eigenvectors_∇F_euler(u::SVector{3,T}, gas) where {T}
    v = u[2] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas)
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
    H = dimensionless_total_enthalpy_density(u, gas)
    r1 = SVector(1.0, v1 - a, v2, H - a * v1)
    r2 = SVector(0.0, 0.0, 1.0, v2)
    r3 = SVector(1.0, v1, v2, 0.5 * (v1 * v1 + v2 * v2))
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
    H = dimensionless_total_enthalpy_density(u, gas)
    r1 = SVector(1.0, v1, v2 - a, H - a * v2)
    r2 = SVector(0.0, 1.0, 0.0, v1)
    r3 = SVector(1.0, v1, v2, 0.5 * (v1 * v1 + v2 * v2))
    r4 = SVector(1.0, v1, v2 + a, H + a * v2)
    return hcat(r1, r2, r3, r4)
end
