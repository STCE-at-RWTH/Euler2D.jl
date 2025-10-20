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

### CHANGE COORDINATE VECTORS FOR VELOCITY SPACE

"""
    scale_velocity_coordinates(u, T)

Multiply the momentum vector of the state `u=[ρ, ρv..., ρE]` by T.
"""
function scale_velocity_coordinates(u, T)
    ρv_new = T * select_middle(u)
    return vcat_ρ_ρv_ρE_preserve_static(u[begin], ρv_new, u[end])
end

"""
    shift_velocity_coordinates(u_star, v0)

Shift the velocity coordinates to a frame moving at `v0`.
Total energy will change due to the new kinetic energy in the new frame.
"""
function shift_velocity_coordinates(u_star, v0)
    ρ = u_star[begin]
    v_star = dimensionless_velocity(u_star)
    v = v_star - v0
    ρe = dimensionless_internal_energy_density(u_star)
    ρv = ρ * v
    return vcat_ρ_ρv_ρE_preserve_static(ρ, ρv, ρe + ρv ⋅ v / 2)
end

"""
    change_velocity_coordinates(u_star, T, v0)

Switch to a new coordinate space for velocity by
switching to a frame moving at `v0` and scaling by `T`.
"""
function change_velocity_coordinates(u_star, T, v0)
    # scale then shift, I think
    u1 = shift_velocity_coordinates(u_star, v0)
    return scale_velocity_coordinates(u1, T)
end

### END OF BASES

### FINITE DIFFS
"""Get an appropriate `ε` for finite differences around `arg`."""
function fdiff_eps(arg::T) where {T<:Real}
    cbrt_eps = cbrt(eps(T))
    h = 2^(round(log2((1 + abs(arg)) * cbrt_eps)))
    return h
end

###

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

### ELEMENTWISE MAXIMUM, MINIMUM, AND EXTREMA
function bcast_max(a, b)
    return max.(a, b)
end

function bcast_min(a, b)
    return min.(a, b)
end
###

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
const _dims_dirs = ((:west, :east), (:south, :north))
