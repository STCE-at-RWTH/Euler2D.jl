n_space_dims(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N
quantity_types(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = (U1, U2, U3)
quantity_types(::Type{ConservedProps{N,T,U1,U2,U3}}) where {N,T,U1,U2,U3} = (U1, U2, U3)
numeric_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = T
numeric_dtype(::Type{ConservedProps{N,T,U1,U2,U3}}) where {N,T,U1,U2,U3} = T

u_array_space_dims(::AbstractArray{T,N}) where {T,N} = N - 1
u_array_space_size(u::AbstractArray{T,N}) where {T,N} = size(u)[2:end]

vcat_ρ_ρv_ρE_preserve_static(u1, u2, u3) = vcat(u1, u2, u3)
vcat_ρ_ρv_ρE_preserve_static(u1, u2::StaticVector{S,T}, u3) where {S,T} =
    SVector{S + 2}(u1, u2..., u3)

function select_middle(u::StaticVector{S,T}) where {S,T}
    idxs = SVector{S - 2}(ntuple(i -> i + 1, S - 2))
    return u[idxs]
end

select_middle(u::AbstractVector) = @view u[2:end-1]

function free_space_dims(N, d)
    ((i + 1 for i ∈ 1:N if i ≠ d)...,)
end

function boundary_velocity_scaling(T, dim, rev)
    return SVector(ntuple(4) do i
        (i == dim + 1) && rev ? -one(T) : one(T)
    end)
end

function split_svector(v)
    N = length(v) ÷ 2
    M = length(v)
    v1, v2 = @inbounds begin
        (SVector{N}(@view v[1:N]), SVector{M - N}(@view v[N+1:M]))
    end
    return v1, v2
end

"""
    flip_velocity(u, dim)

Flip the `dim`th velocity component of `u`. and return a copy.
"""
function flip_velocity(u::ConservedProps{N,T,U1,U2,U3}, dim) where {N,T,U1,U2,U3}
    scaling = SVector(ntuple(i -> i == dim ? -one(T) : one(T), N))
    return ConservedProps(u.ρ, scaling .* u.ρv, u.ρE)
end

merge_values_tuple(arg1, arg2) = (arg1, arg2)
merge_values_tuple(arg1::Tuple, arg2) = (arg1..., arg2)
merge_values_tuple(arg1, arg2::Tuple) = (arg1, arg2...)
merge_values_tuple(arg1::Tuple, arg2::Tuple) = (arg1..., arg2...)

"""
    merge_named_tuples(nt1::NamedTuple{NAMES}, nt2::NamedTuple{NAMES}, nts::NamedTuple{NAMES}...)

Merge the values of the provided named tuples. Will flatten any tuple fields.
"""
function merge_named_tuples(nt1::NamedTuple{NAMES}, nt2::NamedTuple{NAMES}) where {NAMES}
    new_values = ntuple(length(NAMES)) do i
        merge_values_tuple(nt1[NAMES[i]], nt2[NAMES[i]])
    end
    return NamedTuple{NAMES}(new_values)
end

function merge_named_tuples(nt1::NamedTuple{NAMES}, nts::NamedTuple{NAMES}...) where {NAMES}
    return merge_named_tuples(merge_named_tuples(nt1, nts[1]), nts[2:end]...)
end

function _prepend_names(nt::NamedTuple{NAMES}) where {NAMES}
    new_values = ntuple(length(NAMES)) do i
        (NAMES[i], nt[NAMES[i]])
    end
    return NamedTuple{NAMES}(new_values)
end

# named constants
const _dirs_bc_is_reversed = (north = true, south = false, east = false, west = true)
const _dirs_dim = (north = 2, south = 2, east = 1, west = 1)