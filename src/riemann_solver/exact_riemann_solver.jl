"""
  eigenvalues_∇F_euler(u::SVector, gas)

Compute the eigenvalues of the Jacobian of the flux function to the 1-D Euler Equations.
"""
function eigenvalues_∇F_euler(u::SVector{3,T}, gas) where {T}
    v = u[2] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    return SVector(v - a, v..., a)
end

function eigenvectors_∇F_euler(u::SVector{3,T}, gas) where {T}
    v = u[2] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas)
    r1 = SVector(1, v - a, H - v * a)
    r2 = SVector(1, v, v * v / 2)
    r3 = SVector(1, v + a, H + v * a)
    return hcat(r1, r2, r3)
end
