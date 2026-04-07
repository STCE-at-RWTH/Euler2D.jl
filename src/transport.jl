"""
    F_euler(u, gas)

Computes the value of the flux function ``F(u)`` for the Euler equations.
Outputs a matrix with one _column_ for each space dimension.

This implementation will strip out units, and convert down to metric base units in the process. 
"""
function F_euler(u::SVector{S, T}, gas) where {S, T}
    ρv = select_middle(u)
    v = dimensionless_velocity(u)
    P = dimensionless_pressure(u, gas)
    ρv_flux = ρv * v' + I * P
    return vcat(ρv', ρv_flux, (v * (u[end] + P))')
end

"""
    F_euler(u, new_coordinates, gas)

Computes the value of the flux function ``F(u)`` for the Euler eqations,
but converts `u` into a co-ordinate system given by `new_coordinates` first.
"""
function F_euler(u, new_coordinates, gas)
    q = change_space_coordinates(u, new_coordinates)
    return F_euler(q, gas)
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
    F[(begin + 1):(end - 1), :] .= (ρv .* ρv') ./ ρ
    F[(begin + 1):(end - 1), :] += I * P
    c1 = (ρE + P) / ρ
    @. F[end, :] = ρv * c1
    return F
end

function F_euler(u::ConservedProps, gas, scale::EulerEqnsScaling)
    return F_euler(nondimensionalize(u, scale), gas)
end

# why do I still have this
select_space_dim(F_e, dim) = F_e[:, dim]

function ∇F_euler(u, dim, gas)
    return jacobian((x, g, d) -> F_euler(x, g)[:, d], fdiff_backend, u, Constant(gas), Constant(dim))
end

function ∇F_euler(u, gas)
    return stack(map(d -> ∇F_euler(u, d, gas), 1:(length(u) - 2)); dims = 3)
end

function ∇F_euler(u::SVector{N}, gas) where {N}
    jacs = ntuple(N - 2) do i
        ∇F_euler(u, i, gas)
    end
    return _stack_smatrices(jacs...)
end


# do we need the multiple eigenvalues in the middle? I do not know...
"""
    eigenvalues_∇F_euler(u, dims, gas)

Computes the eigenvalues of the Jacobian of the Euler flux function in dimension `dim`.
"""
function eigenvalues_∇F_euler(u, dim, gas)
    v = dimensionless_velocity(u)
    a = dimensionless_speed_of_sound(u, gas)
    return SVector(v[dim] - a, ntuple(Returns(v[dim]), length(v))..., v[dim] + a)
end

eigenvalues_∇F_euler(u, gas) = eigenvalues_∇F_euler(u, 1, gas)
eigenvalues_∇G_euler(u, gas) = eigenvalues_∇F_euler(u, 2, gas)

function eigenvalues_∇F_euler(
        u::ConservedProps{N, T, Q1, Q2, Q3},
        dim,
        gas::CaloricallyPerfectGas,
    ) where {N, T, Q1, Q2, Q3}
    v = ustrip.(ShockwaveProperties._units_v, velocity(u))
    a = ustrip(ShockwaveProperties._units_v, speed_of_sound(u, gas))
    return SVector(v[dim] - a, ntuple(Returns(v[dim]), N)..., v[dim] + a)
end

function right_eigenvectors_∇F_euler(u::SVector{3, T}, gas) where {T}
    v = u[2] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = SVector(1, v - a, H - v * a)
    r2 = SVector(1, v, v * v / 2)
    r3 = SVector(1, v + a, H + v * a)
    return hcat(r1, r2, r3)
end

"""
    right_eigenvectors_∇F_euler(u, gas)

Compute the _right_ eigenvectors of the Jacobian of the x-component of the Euler equations flux.
Ordering of the eigenvectors is the same as in [Shu et al.](https://doi.org/10.1016/0168-9274(92)90066-M).

See also: [`F_euler`](@ref), [`eigenvalues_∇F_euler`](@ref), [`left_eigenvectors_∇F_euler`](@ref)
"""
function right_eigenvectors_∇F_euler(u::SVector{4, T}, gas) where {T}
    v1, v2 = dimensionless_velocity(u)
    q = v1^2 + v2^2
    c = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = @SArray [1.0; v1 - c; v2; H - c * v1]
    r2 = @SArray [0.0; 0.0; 1.0; v2]
    r3 = @SArray [1.0; v1; v2; 0.5 * q]
    r4 = @SArray [1.0; v1 + c; v2; H + c * v1]
    return hcat(r1, r2, r3, r4)
end

"""
    left_eigenvectors_∇F_euler(u, gas)

Compute the _left_ eigenvectors of the Jacobian of the x-component of the Euler equations flux.
Ordering of the eigenvectors is the same as in [Shu et al](https://doi.org/10.1016/0168-9274(92)90066-M).

!!! warning

    There may be a printing error in that reference! ``R^{-1}_{1,2}`` is missing a minus sign.

See also: [`F_euler`](@ref), [`eigenvalues_∇F_euler`](@ref), [`right_eigenvectors_∇F_euler`](@ref)
"""
function left_eigenvectors_∇F_euler(u::SVector{4, T}, gas) where {T}
    v1, v2 = dimensionless_velocity(u)
    q_sqr = v1^2 + v2^2
    c = dimensionless_speed_of_sound(u, gas)
    b1 = (gas.γ - 1) / c^2
    b2 = 0.5 * q_sqr * b1
    l1 = 0.5 * @SArray [b2 + v1 / c ;; -(b1 * v1 + 1 / c) ;; -b1 * v2 ;; b1]
    l2 = @SArray [-v2 ;; 0 ;; 1 ;; 0]
    l3 = @SArray [(1 - b2) ;; b1 * v1 ;; b1 * v2 ;; -b1 ]
    l4 = 0.5 * @SArray [b2 - v1 / c ;; -(b1 * v1 - 1 / c) ;; -b1 * v2 ;; b1]
    return vcat(l1, l2, l3, l4)
end

"""
    right_eigenvectors_∇G_euler(u, gas)

Compute the _right_ eigenvectors of the Jacobian of the y-component of the Euler equations flux.
Ordering of the eigenvectors is the same as in [Shu et al.](https://doi.org/10.1016/0168-9274(92)90066-M).

!!! warning

    There may be a printing error in that reference! ``R^{-1}_{1,2}`` is missing a minus sign.

See also: [`F_euler`](@ref), [`eigenvalues_∇G_euler`](@ref), [`left_eigenvectors_∇G_euler`](@ref)
"""
function right_eigenvectors_∇G_euler(u::SVector{4, T}, gas) where {T}
    v1 = u[2] / u[1]
    v2 = u[3] / u[1]
    a = dimensionless_speed_of_sound(u, gas)
    H = dimensionless_total_enthalpy_density(u, gas) / u[1]
    r1 = SVector(1.0, v1, v2 - a, H - a * v2)
    r2 = SVector(0.0, 1.0, 0.0, v1)
    r3 = SVector(1.0, v1, v2, 0.5 * (v1 * v1 + v2 * v2))
    r4 = SVector(1.0, v1, v2 + a, H + a * v2)
    return mapreduce(normalize, hcat, (r1, r2, r3, r4))
end

function _avg_jacobian_igrand(ϑ, params)
    (uL, uR, gas) = params
    arg = ϑ * uL + (1 - ϑ) * uR
    return jacobian((u, gas) -> F_euler(u, gas)[:, 1], fdiff_backend, arg, Constant(gas))
end

function avg_jacobian_F_euler(qL, qR, gas)
    domain = (0.0, 1.0)
    problem = IntegralProblem(_avg_jacobian_igrand, domain, (qL, qR, gas))
    sol = solve(problem, HCubatureJL())
    @assert SciMLBase.successful_retcode(sol)
    return sol.u
end
