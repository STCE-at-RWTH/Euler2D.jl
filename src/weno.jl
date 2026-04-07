module WENOReconstruction

using LinearAlgebra

using Accessors
using DifferentiationInterface
using StaticArrays
using StaticArrays: sacollect

using Euler2D
using Euler2D: fdiff_backend

export weno_reconstruct_at

# there is a better way to handle this
# (barycentric form)
# but does it play nice with AD?
function lagrange_j(x, j, xs)
    acc_num = one(x)
    acc_denom = one(x)
    for i in eachindex(xs)
        i == j && continue
        acc_num *= (x - xs[i])
        acc_denom *= (xs[j] - xs[i])
    end
    return acc_num / acc_denom
end

function dlagrange_j(x, j, xs)
    return derivative(lagrange_j, fdiff_backend, x, Constant(j), Constant(xs))
end

"""
    eno_stencil_coeffs(x_local, r)
    eno_stencil_coeffs(x_local, r, k)

Compute the ENO stencil coefficients 
for the ENO reconstruction of a function with known cell-averages 
using `r` points.

We assume that ``x = 0`` is the right-most face of the left-most stencil.

This function reproduces the the transpose of the coefficients in *Table 1* in [Jiang & Shu](https://doi.org/10.1006/jcph.1996.0130).
The entries in the returned matrix `a[ℓ, k]` correspond to Jiang & Shu's ``a^r_{k,ℓ}``.

Careful with one-indexing!
"""
function eno_stencil_coeffs(x_local, r)
    # the interpolation point values contain a cumulative sum
    # of the thing we want to interpolate
    # so we scale like this
    # TODO: the minus sign here is sus
    return tril!(ones(r, r)) * -dlagrange_j.(x_local, 1:r, [(-r + k):k for k in 0:(r - 1)]')
end

function eno_stencil_coeffs(x_local, ::Val{R}) where {R}
    prefactor = tril(@SMatrix ones(R, R))
    orders = SVector(ntuple(i -> i, R))
    stencils = ntuple(R) do r
        k = r - 1
        SVector{R + 1}(((-R + k):k)...)
    end |> SVector
    return prefactor * -dlagrange_j.(x_local, orders, stencils')
end

function eno_stencil_coeffs(x_local, r, k)
    # TODO efficiency?
    return eno_stencil_coeffs(x_local, r)[:, k]
end

function weno_optimal_weights(x_local, r)
    small_stencils = eno_stencil_coeffs(x_local, r)
    big_stencil = eno_stencil_coeffs(x_local, 2 * r - 1, r)
    A = fill(zero(x_local), (2 * r - 1, r))
    for i in 1:r
        A[i:(i + (r - 1)), i] = small_stencils[:, i]
    end
    Crk = pinv(A) * big_stencil
    Crk ./= sum(Crk)
    return Crk
end

function weno_optimal_weights(x_local::T, r::Val{R}) where {T <: Number, R}
    if R > 5
        # intervene for speed about here
        return sacollect(SVector{7}, weno_optimal_weights(x_local, R))
    end
    small_stencils = eno_stencil_coeffs(x_local, r)
    big_stencil = eno_stencil_coeffs(x_local, Val{2 * R - 1}(), R)
    A = hcat(
        ntuple(
            (i,) -> begin
                v = @SVector zeros(T, 2 * R - 1)
                for j in 1:R
                    v = setindex(v, small_stencils[j, i], j + i - 1)
                end
                v
            end, R
        )...
    )
    Crk = LowerTriangular(A[SVector(SOneTo(R)...), :]) \ big_stencil[SVector(SOneTo(R)...)]
    # Crk = inv(A[SVector(SOneTo(R)...), :]) * big_stencil[SVector(SOneTo(R)...)]
    return Crk
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
    (r, k, order) = p
    a = unwrap_derivative(x, eno_stencil_coeffs, order, AutoForwardDiff(), Constant(r), Constant(k))
    return a .* a'
end

function _weno_smoothness_ℓ(r, k, ℓ)
    _igral = IntegralProblem(_weno_smoothness_igrand, -1.0, 0.0, (r, k, ℓ))
    _res = solve(_igral, HCubatureJL())
    return _res.u
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

struct WENOReconstructor{ORDER, T, COEFFS, WEIGHTS, SMOOTHNESS}
    x_local::T
    coeffs::COEFFS
    optimal_weights::WEIGHTS
    smoothness_measure::SMOOTHNESS
end

function WENOReconstructor{ORDER}(x_local::T) where {ORDER, T}
    c = eno_stencil_coeffs(x_local, Val(ORDER))
    opt_w = weno_optimal_weights(x_local, Val(ORDER))
    ks = SVector(SOneTo(ORDER)...)
    ℓs = SVector(SOneTo(ORDER - 1)...)
    sm = _weno_smoothness_ℓ.(Val(ORDER), ks, ℓs')
    return WENOReconstructor{ORDER, T, typeof.((c, opt_w, sm))...}(x_local, c, opt_w, sm)
end

reconstruction_order(::WENOReconstructor{R}) where {R} = R
reconstruction_point(r::WENOReconstructor{R}) where {R} = r.x_local
reconstruction_stencil(r::WENOReconstructor) = r.coeffs
reconstruction_stencil(r::WENOReconstructor, k) = r.coeffs[:, k]

weno_optimal_weights(rc::WENOReconstructor, _) = rc.optimal_weights
weno_optimal_weights(rc::WENOReconstructor) = rc.optimal_weights

_weno_smoothness_ℓ(rc::WENOReconstructor, k, ℓ) = rc.smoothness_measure[k, ℓ]
weno_smoothness(u_Sk, rc::WENOReconstructor{R}, k) where {R} = _weno_smoothness(u_Sk, rc, k, SVector(SOneTo(R - 1)...))

function weno_reconstruct_at(rc::WENOReconstructor, u)
    r = reconstruction_order(rc)
    ISk = [weno_smoothness(@view(u[idx:(idx + r - 1)]), rc, idx) for idx in 1:r]
    w_k = weno_stencil_weights(rc, ISk, reconstruction_order(rc))
    acc = zero(eltype(u))
    for k in 1:reconstruction_order(rc)
        acc += w_k[k] * (reconstruction_stencil(rc, k)' * @view(u[k:(k + r - 1)]))
    end
    return acc
end

function weno_reconstruct_at(rc::WENOReconstructor{R}, u::SVector{NU, UT}) where {R, NU, UT}
    @assert isodd(NU) "Need an odd stencil width."
    @assert (NU == 2 * R - 1) "Reconstruction doesn't match provided values of u."
    u_stencils = ntuple(R) do idx
        offset = idx - 1
        return u[SVector(SOneTo(R)...) .+ offset]
    end |> SVector
    ISk = ntuple(R) do idx
        weno_smoothness(u_stencils[idx], rc, idx)
    end |> SVector
    w_k = weno_stencil_weights(rc, ISk, Val(R))
    acc = zero(UT)
    for k in 1:R
        acc += w_k[k] * (reconstruction_stencil(rc, k)' * u_stencils[k])
    end
    return acc
end


function weno_reconstruct_at(x_local, u)
    if !(-1.0 < x_local < 0.0)
        @warn "Smoothness measure not relevant here;" x = x_local
    end
    r = div(length(u) + 1, 2)
    # TODO don't need this temporary
    u_stencils = [@view u[idx:(idx + r - 1)] for idx in 1:r]
    a_kl = eno_stencil_coeffs(x_local, r)
    ISk = [weno_smoothness(_u, r, idx) for (idx, _u) in enumerate(u_stencils)]
    w_k = weno_stencil_weights(x_local, ISk, r)
    acc = zero(eltype(u))
    for k in 1:r
        acc += w_k[k] * ((a_kl[:, k])' * u_stencils[k])
    end
    return acc
end

function weno_reconstruct_at(x_local, u::SVector{NU, UT}) where {NU, UT}
    @assert isodd(NU) "Need an odd stencil width."
    if !(-1.0 < x_local < 0.0)
        @warn "Smoothness measure not relevant here;" x = x_local
    end
    R = div(NU + 1, 2)
    u_stencils = ntuple(R) do idx
        offset = idx - 1
        slice = ntuple(i -> offset + i, R) |> SVector
        return u[slice]
    end |> SVector
    a_kl = eno_stencil_coeffs(x_local, Val(R))
    ISk = ntuple(R) do idx
        weno_smoothness(u_stencils[idx], Val(R), idx)
    end |> SVector
    w_k = weno_stencil_weights(x_local, ISk, Val(R))
    acc = zero(UT)
    for k in 1:R
        acc += w_k[k] * (a_kl[:, k]' * u_stencils[k])
    end
    return acc
end

function weno_reconstruct_at(x, u, xs)
    @assert length(xs) == length(u)
    dx = xs[2] - xs[1]
    r = div(length(xs) + 1, 2)
    x_loc = ((x - xs[r])) / dx - 0.5
    return weno_reconstruct_at(x_loc, u)
end

end
