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

function roe_parameter_vector(u::SVector{N}, gas) where {N}
    ρH = dimensionless_total_enthalpy_density(u, gas)
    w = setindex(u, ρH, N) / sqrt(u[1])
    return w
end

"""
    roe_parameter_mean(uL, uR, gas)

Get the arithmetic mean of the Roe parameter vectors ``w_L`` and ``w_R``
from the state vectors ``u_L`` and ``u_R``. 
"""
function roe_parameter_mean(uL, uR, gas)
    wL = roe_parameter_vector(uL, gas)
    wR = roe_parameter_vector(uR, gas)
    return 0.5 * (wL + wR)
end

"""
    roe_mean_q_sqr(w̄)

Compute ``q^2 = v⋅v`` in the Roe-average.
"""
function roe_mean_q_sqr(w_roe)
    v_roe = dimensionless_velocity(w_roe)
    return v_roe' * v_roe
end

roe_mean_q_sqr(uL, uR, gas) = roe_mean_q_sqr(roe_parameter_mean(uL, uR, gas))

"""
    roe_mean_c_sqr(w̄, gas)

Compute the speed of sound squared in the Roe-average.
"""
function roe_mean_c_sqr(w_roe, gas)
    H_roe = last(w_roe) / first(w_roe)
    qsqr_roe = roe_mean_q_sqr(w_roe)
    return (gas.γ - 1) * (H_roe - 0.5 * qsqr_roe)
end

roe_mean_c_sqr(uL, uR, gas) = roe_mean_c_sqr(roe_parameter_mean(uL, uR, gas), gas)

"""
    roe_B_tilde(w̄, gas)

Equation 14 in [_Roe_](https://doi.org/10.1016/0021-9991(81)90128-5). 
"""
function roe_B_tilde(w̄, gas)
    length(w̄) == 4 || throw(ArgumentError("Roe parameter vector needs length 4."))
    a = (gas.γ - 1) / (gas.γ)
    b = inv(gas.γ)
    c1 = @SArray [2 * w̄[1]; w̄[2]; w̄[3]; b * w̄[4]]
    c2 = @SArray [0; w̄[1]; 0; a * w̄[2]]
    c3 = @SArray [0; 0; w̄[1]; a * w̄[3]]
    c4 = @SArray [0; 0; 0; b * w̄[1]]
    return hcat(c1, c2, c3, c4)
end

"""
    roe_Cx_tilde(w̄, gas)

Equation 16 in [_Roe_](https://doi.org/10.1016/0021-9991(81)90128-5). 
Used to compute a jump in the x-direction Euler flux from the average parameter vector ``w̄``.
"""
function roe_Cx_tilde(w_roe, gas)
    length(w_roe) == 4 || throw(ArgumentError("Roe parameter vector needs length 4."))
    a = (gas.γ - 1) / (gas.γ)
    b = (gas.γ + 1) / (gas.γ)
    c1 = @SArray [w_roe[2]; a * w_roe[4]; 0; 0;]
    c2 = @SArray [w_roe[1]; b * w_roe[2]; w_roe[3]; w_roe[4]]
    c3 = @SArray [0; -a * w_roe[3]; w_roe[2]; 0]
    c4 = @SArray [0; a * w_roe[1]; 0; w_roe[2]]
    return hcat(c1, c2, c3, c4)
end

function roe_average_F_euler(uL, uR, gas)
    w̄ = roe_parameter_mean(uL, uR, gas)
    return roe_average_F_euler(w̄, gas)
end

function roe_average_F_euler(w_roe, gas)
    return roe_Cx_tilde(w_roe, gas) * inv(roe_B_tilde(w_roe, gas))
end

function right_eigenvectors_roe_average_F_euler_fleischmann(w_roe, gas)
    @assert length(w_roe) == 4 "Only implemented for 2-d systems"
    v1_roe, v2_roe = dimensionless_velocity(w_roe)
    H_roe = last(w_roe) / first(w_roe)
    csqr_roe = roe_mean_c_sqr(w_roe, gas)
    c_roe = sqrt(csqr_roe)
    b1 = inv(H_roe - 0.5 * roe_mean_q_sqr(w_roe))
    r1 = @SVector [1; v1_roe - c_roe; v2_roe; H_roe - c_roe * v1_roe]
    r2 = @SVector [b1; b1 * v1_roe; b1 * v2_roe; b1 * H_roe - 1]
    r3 = @SVector [0; 0; 1; v2_roe;]
    r4 = @SVector [1; v1_roe + c_roe; v2_roe; H_roe + c_roe * v1_roe]
    return hcat(r1, r2, r3, r4)
end

"""
    right_eigenvectors_roe_average_F_euler(w̄, gas)

Compute the _right_ eigenvectors of the roe average matrix for the ``x``-direction
Euler flux as done in [Roe](https://doi.org/10.1016/0021-9991(81)90128-5) with the scaling from
[Fleischmann et al.](https://doi.org/10.1016/j.compfluid.2019.04.004)

See also: [`left_eigenvectors_roe_average_F_euler`](@ref), [`right_eigenvectors_∇F_euler`](@ref)
"""
function right_eigenvectors_roe_average_F_euler(w_roe, gas)
    @assert length(w_roe) == 4 "Only implemented for 2-d systems"
    v1_roe, v2_roe = dimensionless_velocity(w_roe)
    H_roe = last(w_roe) / first(w_roe)
    csqr_roe = roe_mean_c_sqr(w_roe, gas)
    qsqr_roe = roe_mean_q_sqr(w_roe)
    c_roe = sqrt(csqr_roe)
    r1 = @SVector [1; v1_roe - c_roe; v2_roe; H_roe - c_roe * v1_roe]
    r2 = @SVector [1; v1_roe; v2_roe; 0.5 * qsqr_roe]
    r3 = @SVector [0; 0; 1; v2_roe;]
    r4 = @SVector [1; v1_roe + c_roe; v2_roe; H_roe + c_roe * v1_roe]
    return hcat(r1, r2, r3, r4)
end

right_eigenvectors_roe_average_F_euler(uL, uR, gas) = right_eigenvectors_roe_average_F_euler(roe_parameter_mean(uL, uR, gas), gas)

"""
    left_eigenvectors_roe_average_F_euler(w̄, gas)

Compute the _left_ eigenvectors of the roe average matrix for the ``x``-direction
Euler flux as done in [Fleischmann et al.](https://doi.org/10.1016/j.compfluid.2019.04.004)

See also [`right_eigenvectors_roe_average_F_euler`](@ref), [`left_eigenvectors_∇F_euler`](@ref)
"""
function left_eigenvectors_roe_average_F_euler(w_roe, gas)
    @assert length(w_roe) == 4 "Only implemented for 2-d systems"
    v1_roe, v2_roe = dimensionless_velocity(w_roe)
    H_roe = last(w_roe) / first(w_roe)
    csqr_roe = roe_mean_c_sqr(w_roe, gas)
    c_roe = sqrt(csqr_roe)
    qsqr_roe = roe_mean_q_sqr(w_roe)
    b1 = inv(H_roe - 0.5 * qsqr_roe)
    b2 = 0.5 * qsqr_roe * b1
    l1 = 0.5 * @SArray [b2 + v1_roe / c_roe;; -(b1 * v1_roe + 1 / c_roe);; -b1 * v2_roe;; b1]
    l3 = @SArray [-v2_roe;; 0;; 1;; 0]
    l2 = @SArray [(1 - b2);; b1 * v1_roe;; b1 * v2_roe;; -b1]
    l4 = 0.5 * @SArray [b2 - v1_roe / c_roe;; -(b1 * v1_roe - 1 / c_roe);; -b1 * v2_roe;; b1]
    return vcat(l1, l2, l3, l4)
end

left_eigenvectors_roe_average_F_euler(uL, uR, gas) = left_eigenvectors_roe_average_F_euler(roe_parameter_mean(uL, uR, gas), gas)

"""
    roe_average_eigenvalues(w̄, dim, gas)

Find the eigenvalues of the Roe average matrix for the ``dim``-th direction of the Euler flux. 
"""
function roe_average_eigenvalues(w_roe, dim::Number, gas)
    v_roe = dimensionless_velocity(w_roe)
    H_roe = w_roe[end] / w_roe[1]
    a = sqrt((gas.γ - 1) * (H_roe - (v_roe ⋅ v_roe) / 2))
    return if v_roe isa SVector
        vcat(SVector(v_roe[dim] - a), v_roe[dim] * ones(typeof(v_roe)), SVector(v_roe[dim] + a))
    else
        vcat(v_roe[dim] - a, fill(v_roe[dim], length(v_roe)), v_roe[dim] + a)
    end
end

roe_average_eigenvalues(uL, uR, dim::Number, gas) = roe_average_eigenvalues(roe_parameter_mean(uL, uR, gas), dim, gas)
roe_average_eigenvalues(uL, uR, gas) = roe_average_eigenvalues(roe_parameter_mean(uL, uR, gas), 1, gas)
roe_average_eigenvalues(w_roe, gas) = roe_average_eigenvalues(w_roe, 1, gas)
