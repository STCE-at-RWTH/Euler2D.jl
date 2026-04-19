module WENOReconstruction

using LinearAlgebra

using Accessors
using DifferentiationInterface
using StaticArrays
using StaticArrays: sacollect

using Euler2D
using Euler2D: fdiff_backend

export WENOReconstructor
export weno_reconstruct_at

struct WENOReconstructor{ORDER, T, XFACES, COEFFS, WEIGHTS, SMOOTHNESS}
    x::T
    faces::XFACES
    coeffs::COEFFS
    optimal_weights::WEIGHTS
    smoothness_measure::SMOOTHNESS
end

function WENOReconstructor{ORDER}(x::T, x_faces::XF, c::C, opt_w::W, sm::SM) where {ORDER, T, XF, C, W, SM}
    return WENOReconstructor{ORDER, T, XF, C, W, SM}(x, x_faces, c, opt_w, sm)
end

function WENOReconstructor{ORDER}(x_local::T) where {ORDER, T}
    return WENOReconstructor{ORDER}(x_local, range(; start = -one(T) * ORDER, length = 2 * ORDER, step = one(T)))
end

function WENOReconstructor{ORDER}(x::T, xs::AbstractVector) where {ORDER, T}
    @assert length(xs) == 2 * ORDER
    c = eno_stencil_coeffs(x, xs, Val(ORDER))
    opt_w = weno_optimal_weights(x, xs, Val(ORDER))
    sm = zeros(SMatrix{ORDER, ORDER, T, ORDER * ORDER}, (ORDER, ORDER - 1))
    # I think this is right even if we have a left-facing set of stencils
    for k in axes(sm, 1), ℓ in axes(sm, 2)
        sm[k, ℓ] = _weno_smoothness_ℓ(xs[ORDER], xs[ORDER + 1], xs, Val(ORDER), k, ℓ)
    end
    return WENOReconstructor{ORDER}(x, xs, c, opt_w, sm)
end

reconstruction_order(::WENOReconstructor{R}) where {R} = R
reconstruction_point(r::WENOReconstructor{R}) where {R} = r.x
reconstruction_face_nodes(r::WENOReconstructor{R}) where {R} = r.faces

reconstruction_stencil(r::WENOReconstructor) = r.coeffs
reconstruction_stencil(r::WENOReconstructor, k) = r.coeffs[:, k]

weno_optimal_weights(rc::WENOReconstructor, _) = rc.optimal_weights
weno_optimal_weights(rc::WENOReconstructor) = rc.optimal_weights

_weno_smoothness_ℓ(rc::WENOReconstructor, k, ℓ) = rc.smoothness_measure[k, ℓ]
weno_smoothness(u_Sk, rc::WENOReconstructor{R}, k) where {R} = _weno_smoothness(u_Sk, rc, k, SVector(SOneTo(R - 1)...))

function weno_reconstruct_at(rc::WENOReconstructor, u::AbstractVector{UT}) where {UT}
    R = reconstruction_order(rc)
    ISk = ntuple(R) do idx
        slice = idx:(idx + R - 1)
        weno_smoothness(@view(u[slice]), rc, idx)
    end |> SVector
    w_k = weno_stencil_weights(rc, ISk, R)
    acc = zero(eltype(u))
    for k in 1:R
        acc += w_k[k] * (reconstruction_stencil(rc, k)' * @view(u[k:(k + R - 1)]))
    end
    return acc
end

function _swap_inner_vector(v_of_v)
    L = eachindex(first(v_of_v))
    return map(L isa SOneTo ? SVector(L...) : L) do i
        getindex.(v_of_v, i)
    end
end

function weno_reconstruct_at(rc_plus::WENOReconstructor{ORDER}, rc_minus::WENOReconstructor{ORDER}, u, L, Λ, R) where {ORDER}
    w = map(_u -> L * _u, u)
    characteristic_fields = _swap_inner_vector(w)
    w_weno = map(Λ, characteristic_fields) do λ_i, w_i
        if λ_i > 0
            return weno_reconstruct_at(rc_plus, @view w_i[begin:(end - 1)])
        end
        return weno_reconstruct_at(rc_minus, @view w_i[(begin + 1):end])
    end
    return R * w_weno
end

function weno_reconstruct_at(x, u, xs::AbstractRange)
    @assert length(xs) == length(u)
    dx = step(xs)
    r = div(length(xs) + 1, 2)
    x_loc = ((x - xs[r])) / dx - 0.5
    return weno_reconstruct_at(x_loc, u)
end

# there is a better way to handle this
# (barycentric form)
# but does it play nice with AD?
function lagrange_j(x, j, x_nodes)
    acc_num = one(x)
    acc_denom = one(x)
    for i in eachindex(x_nodes)
        if i == j
            continue
        end
        acc_num *= (x - x_nodes[i])
        acc_denom *= (x_nodes[j] - x_nodes[i])
    end
    return acc_num / acc_denom
end

# turns out that just using DI here is faster...
# there may be accuracy problems when x is close to one of the x_nodes
function dlagrange_j(x, j, x_nodes)
    return derivative(lagrange_j, fdiff_backend, x, Constant(j), Constant(x_nodes))
end

_one_to(r) = Base.OneTo(r)
_one_to(::Val{R}) where {R} = SVector(ntuple(i -> i, R))

_eno_cell_avgs_scaling(r) = tril!(ones(r, r))
_eno_cell_avgs_scaling(::Val{R}) where {R} = tril(@SMatrix ones(R, R))
_eno_cell_avgs_scaling(r, k) = map(i -> i ≤ k ? 1 : 0, _one_to(r))
_eno_cell_avgs_scaling(x_nodes::AbstractRange, r) = step(x_nodes) * _eno_cell_avgs_scaling(r)
_eno_cell_avgs_scaling(x_nodes::AbstractRange, r, k) = step(x_nodes) * _eno_cell_avgs_scaling(r, k)
_eno_cell_avgs_scaling(x_nodes::AbstractVector{<:Real}, r) = Diagonal(diff(x_nodes)) * _eno_cell_avgs_scaling(r)
_eno_cell_avgs_scaling(x_nodes::AbstractVector{<:Real}, r, k) = Diagonal(diff(x_nodes)) * _eno_cell_avgs_scaling(r, k)

function _eno_local_stencils(x_nodes::AbstractVector, r)
    return [x_nodes[(begin + k):(begin + k + r)] for k in 0:(r - 1)]
end

function _eno_local_stencils(x_nodes::AbstractVector, ::Val{R}) where {R}
    slices = ntuple(R) do i
        _one_to(Val(R + 1)) .+ (i - 1)
    end |> SVector
    return map(idxs -> x_nodes[idxs], slices)
end

_eno_local_stencils(x_nodes::AbstractVector, r, k) = x_nodes[_one_to(r + 1) .+ (k - 1)]
function _eno_local_stencils(x_nodes::AbstractVector, ::Val{R}, k) where {R}
    idxs = _one_to(Val(R + 1)) .+ (k - 1)
    return x_nodes[idxs]
end
function _eno_local_stencils(x_nodes::AbstractVector, r::Val{R}, ::Val{K}) where {R, K}
    return _eno_local_stencils(x_nodes, r, K)
end

_eno_local_stencils(r) = _eno_local_stencils((-r):(r - 1), r)
_eno_local_stencils(r::Val{R}) where {R} = _eno_local_stencils((-R):(R - 1), r)
_eno_local_stencils(r, k) = _eno_local_stencils((-r):(r - 1), r, k)
_eno_local_stencils(r::Val{R}, k) where {R} = _eno_local_stencils((-R):(R - 1), r, k)

# the interpolation point values contain a cumulative sum
# of the thing we want to interpolate
# so we scale like this to capture the cumsum and the Δx values

function _eno_stencil_coeffs(x, stencils::AbstractVector, max_order)
    # TODO: the minus sign here is sus
    # return _eno_cell_avgs_scaling(max_order) * -dlagrange_j.(x, _one_to(max_order), stencils')
    return mapreduce(hcat, stencils) do stencil
        scaling = _eno_cell_avgs_scaling(stencil, max_order)
        dℓ_j = -dlagrange_j.(x, _one_to(max_order), (stencil,))
        return scaling * dℓ_j
    end
end

"""
    eno_stencil_coeffs(x, x_nodes, r)

Compute the ENO stencil coefficients for the ENO reconstruction of 
a function with known cell-averages using stencils with `r` cells. The nodes `x_nodes`
are the values of `x` at each face of the cells. Hence, there should be ``2r`` of them,
starting from the left-most face of the left-most stencil and going all the way to the
right-most face of the right-most stencil. The right-most face of the left-most stencil
is at the `r+1`-th value in `x_nodes`.

This function reproduces the the transpose of the coefficients in *Table 1* in 
[Jiang & Shu](https://doi.org/10.1006/jcph.1996.0130). The entries in the returned 
matrix `a[ℓ, k]` correspond to Jiang & Shu's ``a^r_{k,ℓ}``.

    eno_stencil_coeffs(x_local, r)

Automatically generates the stencils as if the grid points were ``-(r+1)`` to ``r``. 
Treats the right-most face of the left-most stencil as ``x=0``.

    eno_stencil_coeffs(x_local, r, k)
    eno_stencil_coeffs(x, x_nodes, r, k)

Get the `k`-th stencil coefficients out of the matrix above. (equal to the `k`-th column.)
"""
function eno_stencil_coeffs(x::Real, x_nodes::AbstractVector{<:Real}, r)
    stencils = _eno_local_stencils(x_nodes, r)
    return _eno_stencil_coeffs(x, stencils, r)
end

function eno_stencil_coeffs(x::Real, x_nodes::AbstractVector{<:Real}, r, k)
    stencil = SVector((_eno_local_stencils(x_nodes, r, k),))
    return _eno_stencil_coeffs(x, stencil, r)
end

function eno_stencil_coeffs(x_local::Real, r)
    stencils = _eno_local_stencils(r)
    return _eno_stencil_coeffs(x_local, stencils, r)
end

function eno_stencil_coeffs(x_local::Real, r, k)
    stencil = SVector((_eno_local_stencils(r, k),))
    return _eno_stencil_coeffs(x_local, stencil, r)
end

# the LowerTriangular method allocates more than it ought to? I am baffled.
function _direct_fwd_sub_lt(A, b)
    _, N = size(A)
    v = similar(A, N)
    for i in eachindex(axes(A, 1))
        acc = zero(eltype(v))
        for j in 1:(i - 1)
            acc += A[i, j] * v[j]
        end
        v[i] = (b[i] - acc) / A[i, i]
    end
    return v
end

function _solve_for_stencil(small_stencil, big_stencil, r)
    A = zeros(eltype(small_stencil), 2 * r - 1, r)
    for i in 1:r
        A[i:(i + (r - 1)), i] = small_stencil[:, i]
    end
    Crk = _direct_fwd_sub_lt(@view(A[1:r, :]), big_stencil)
    Crk .*= inv(sum(Crk))
    return Crk
end

"""
    weno_optimal_weights(x_local, r)

Compute the optimal weights for each of the stencils ``S_k`` for scheme order `r` 
from the fact that the coefficients of ``S_r`` in a scheme of order ``2r-1`` can be constructed
from the stencils of a scheme of order ``r``. 

See also: [`eno_stencil_coeffs`](@ref)
"""
function weno_optimal_weights(x_local, r)
    small_stencils = eno_stencil_coeffs(x_local, r)
    big_stencil = eno_stencil_coeffs(x_local, 2 * r - 1, r)
    return _solve_for_stencil(small_stencils, big_stencil, r)
end

function weno_optimal_weights(x_local::T, r::Val{R}) where {T <: Number, R}
    small_stencils = eno_stencil_coeffs(x_local, r)
    big_stencil = eno_stencil_coeffs(x_local, Val(2 * R - 1), r)
    return sacollect(SVector{R}, _solve_for_stencil(small_stencils, big_stencil, R))
end

function weno_optimal_weights(x, x_nodes::AbstractVector, r)
    small_stencils = eno_stencil_coeffs(x, x_nodes, r)
    big_stencil = _eno_stencil_coeffs(x, SVector((x_nodes,)), 2 * r - 1)
    return _solve_for_stencil(small_stencils, big_stencil, r)
end

function weno_optimal_weights(x, x_nodes::AbstractVector, r::Val{R}) where {R}
    small_stencils = eno_stencil_coeffs(x, x_nodes, r)
    big_stencil = _eno_stencil_coeffs(x, SVector((x_nodes,)), Val(2 * R - 1))
    return _solve_for_stencil(small_stencils, big_stencil, R)
end

function weno_stencil_weights(x_local_or_reconstructor, ISk, r, ε, p)
    C = weno_optimal_weights(x_local_or_reconstructor, r)
    alpha_k = map(C, ISk) do c, isk
        return c * inv((ε + isk)^p)
    end
    w_k = alpha_k / sum(alpha_k)
    return w_k
end

function weno_stencil_weights(x_local, ISk, r)
    return weno_stencil_weights(x_local, ISk, r, 1.0e-6, 2)
end

"""
    eno3_stencil_coeffs()

The ENO-3 stencil coefficients, as presented in "Efficient Implementation of Weighted ENO Schemes".

For reconstruction at the cell face.
"""
function eno3_stencil_coeffs()
    return @SArray [
        1 / 3; -1 / 6; 1 / 3;;
        -7 / 6; 5 / 6; 5 / 6;;
        11 / 6; 1 / 3; -1 / 6;;
    ]
end

function weno3_optimal_weights()
    return @SVector [1 / 10, 6 / 10, 3 / 10]
end

"""
    eno3_stencil_coeffs(k)

The ENO-3 stencil coefficients for stencil `k`. `k` ranges from ``1`` to ``r``.

For reconstruction at the cell face.
"""
function eno3_stencil_coeffs(k)
    return eno3_stencil_coeffs()[k, :]
end

function weno3_smoothness_coeffs()
    return @SArray [
        1;-4; 3;;
        1;-2; 1;;;
        1; 0;-1;;
        1;-2; 1;;;
        3;-4; 1;;
        1;-2; 1
    ]
end

function weno3_smoothness_coeffs(k)
    return weno3_smoothness_coeffs()[:, :, k]
end

function weno3_smoothness_coeffs(ℓ, k)
    return weno3_smoothness_coeffs()[:, ℓ, k]
end

function weno3_smoothness_prefactors()
    return @SVector [1 / 4, 13 / 12]
end

function weno3_smoothness_prefactors(ℓ)
    return weno3_smoothness_prefactors()[ℓ]
end

using Euler2D: opnorm2_sqr

function weno3_smoothness(u_Sk, k)
    return mapreduce(+, SOneTo(2)) do ℓ
        dq_k = weno3_smoothness_coeffs(ℓ, k)' * u_Sk
        return weno3_smoothness_prefactors(ℓ) * opnorm2_sqr(dq_k)
    end
end

using Integrals
using Euler2D: unwrap_derivative

function _weno_smoothness_igrand(x, p)
    (deriv_order, h, ctx...) = p
    a = h^(2 * deriv_order - 1) * unwrap_derivative(x, eno_stencil_coeffs, deriv_order, AutoForwardDiff(), Constant.(ctx)...)
    return a .* a'
end

function _weno_smoothness_ℓ(r, k, ℓ)
    _igral = IntegralProblem(_weno_smoothness_igrand, -1.0, 0.0, (ℓ, 1.0, r, k))
    _res = solve(_igral, HCubatureJL())
    return _res.u
end

function _weno_smoothness_ℓ(xL, xR, x_nodes, r, k, ℓ)
    h = xR - xL
    _igral = IntegralProblem(_weno_smoothness_igrand, xL, xR, (ℓ, h, x_nodes, r, k))
    _res = solve(_igral, HCubatureJL())
    return _res.u
end

function _weno_smoothness_ℓ(::Val{3}, k, ℓ)
    a = weno3_smoothness_coeffs(ℓ, k)
    return weno3_smoothness_prefactors(ℓ) * (a .* a')
end

function _weno_smoothness(u_Sk, r, k, ℓs)
    return sum(ℓs) do ℓ
        A = _weno_smoothness_ℓ(r, k, ℓ)
        return opnorm2_sqr(u_Sk, A)
    end
end

function weno_smoothness(u_Sk, r, k)
    return _weno_smoothness(u_Sk, r, k, Base.OneTo(r - 1))
end

function weno_smoothness(u_Sk, r::Val{R}, k) where {R}
    return _weno_smoothness(u_Sk, r, k, SOneTo(R - 1))
end

function weno3_stencil_weights(ISk, ε, p)
    C = weno3_optimal_weights()
    alpha_k = @. C * inv((ε + ISk)^p)
    w_k = alpha_k / sum(alpha_k)
    return w_k
end

function weno3_stencil_weights(ISk)
    return weno3_stencil_weights(ISk, 1.0e-6, 2)
end


end
