n_space_dims(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N
quantity_types(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = (U1, U2, U3)
quantity_types(::Type{ConservedProps{N,T,U1,U2,U3}}) where {N,T,U1,U2,U3} = (U1, U2, U3)
numeric_dtype(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = T
numeric_dtype(::Type{ConservedProps{N,T,U1,U2,U3}}) where {N,T,U1,U2,U3} = T

u_array_space_dims(::AbstractArray{T,N}) where {T,N} = N - 1
u_array_space_size(u::AbstractArray{T,N}) where {T,N} = size(u)[2:end]

vcat_ρ_ρv_ρE_preserve_static(u1, u2, u3) = vcat(u1, u2, u3)
function vcat_ρ_ρv_ρE_preserve_static(u1, u2::SVector{S,T}, u3) where {S,T}
    return SVector{S + 2}(u1, u2..., u3)
end

### RIPPED FROM PLANEPOLYGONS
### IF THE CUT-CELLS METHOD EVER GETS IMPLEMENTED
### MAYBE WE CAN EXPAND THE INTERFACE IN THE OTHER PACKAGE

"""
  change_of_basis_matrix(A, B)

Get the matrix ``T_{AB}`` to change basis from coordinates ``A`` to ``B``. 

``x_B = Tx_a``.
"""
function change_of_basis_matrix(A, B)
    return inv(B) * A
end

"""
  orthonormal_basis(x)

Get an orthonormal basis from a choice of axis ``e_1``.
"""
function orthonormal_basis(x)
    @assert length(x) == 2
    x̂ = SVector{2}(normalize(x)...)
    ŷ = SVector(-x̂[2], x̂[1])
    return hcat(x̂, ŷ)
end

"""
  change_basis(p, A, B)

Put the vector ``p`` in basis ``A`` into basis ``B``.
"""
function change_basis(p, A, B)
    return change_of_basis_matrix(A, B) * p
end

"""
  change_basis(p, B)

Put the vector ``p`` in the standard Cartesian basis into basis B.
"""
change_basis(p, B) = change_basis(p, I, B)

function project_state_to_normal(u::SVector{S,T}, n) where {S,T}
    ρv_n = select_middle(u) ⋅ n
    return SVector(u[1], ρv_n, zeros(SVector{S - 3,T})..., u[end])
end

function project_state_to_normal(u, n)
    return vcat(u[1], select_middle(u) ⋅ n, zeros(eltype(u), length(u - 3)), u[end])
end

"""
  apply_coordinate_tform(u, T)

Multiply the momentum vector of the state `u=[ρ, ρv..., ρE]` by T.
"""
function apply_coordinate_tform(u, T)
    ρv_new = T * select_middle(u)
    return vcat_ρ_ρv_ρE_preserve_static(u[1], ρv_new, u[end])
end

### END OF BASES

### HORRIBLE BACKUP DICT THINGY
"""
  BackupDict{K, V}

A wrapper around two Dicts that will try to look up values in
`primary` before `secondary`.
"""
struct BackupDict{K,V} #<: AbstractDict{K,V}
    primary::Dict{K,V}
    secondary::Dict{K,V}
end

function Base.getindex(bdict::BackupDict, key)
    return get(() -> bdict.secondary[key], bdict.primary, key)
end

### END HORRIBLE BACKUP DICT THINGY

function free_space_dims(N, d)
    ((i + 1 for i ∈ 1:N if i ≠ d)...,)
end

function boundary_velocity_scaling(T, dim, rev)
    return SVector(ntuple(4) do i
        (i == dim + 1) && rev ? -one(T) : one(T)
    end)
end

"""
  split_svector(v)

Splits the SVector `v` of length `N` into two SVectors `v[1:N÷2]` and `v[(N÷2)+1:N]`.
"""
function split_svector(v)
    N = length(v) ÷ 2
    M = length(v)
    v1, v2 = @inbounds begin
        (SVector{N}(@view v[1:N]), SVector{M - N}(@view v[N+1:M]))
    end
    return v1, v2
end

"""
  select_middle(u)

Select the middle of a vector `u`, returning a copy of `u[2:end-1]` if `u` is an `SVector`, otherwise a view of `u[2:end-1]`.
"""
function select_middle(u::StaticVector{S,T}) where {S,T}
    idxs = SVector{S - 2}(ntuple(i -> i + 1, S - 2))
    return u[idxs]
end
select_middle(u::AbstractVector) = @view u[2:end-1]

ncols_smatrix(::SMatrix{M,N,T,L}) where {M,N,T,L} = N
ncols_smatrix(::Type{SMatrix{M,N,T,L}}) where {M,N,T,L} = N

"""
    flip_velocity(u, dim)

Flip the `dim`th velocity component of `u`. and return a copy. 
"""
function flip_velocity(u::ConservedProps{N,T,U1,U2,U3}, dim) where {N,T,U1,U2,U3}
    scaling = SVector(ntuple(i -> i == dim ? -one(T) : one(T), N))
    return ConservedProps(u.ρ, scaling .* u.ρv, u.ρE)
end

function flip_velocity(u::SVector{N,T}, dim) where {N,T}
    # momentum density vector is stored in u[2] and u[3]
    # or u[2:end-1] for higher-dimensional problems (not that we care)
    scaling = SVector(ntuple(i -> i == 1 + dim ? -one(T) : one(T), N))
    return u .* scaling
end

function flip_velocity(u::SMatrix{M,N,T,L}, dim) where {M,N,T,L}
    scaling = SDiagonal(ntuple(i -> i == 1 + dim ? -one(T) : one(T), M))
    return scaling * u
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

#BUG this allocates
# WHY DOES THIS ALLOCATE
function _prepend_names(nt::NamedTuple{NAMES}) where {NAMES}
    new_values = ntuple(length(NAMES)) do i
        (NAMES[i], nt[NAMES[i]])
    end
    return NamedTuple{NAMES}(new_values)
end

# named constants
const _dirs_bc_is_reversed = (north = true, south = false, east = false, west = true)
const _dirs_dim = (north = 2, south = 2, east = 1, west = 1)

function edge_points(center, extent, θ_n)
    t = extent / 2
    v = SVector(normal_vec[2], -normal_vec[1])

    # HOW DO I DO THIS
    # HOW

end
