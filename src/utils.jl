n_space_dims(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N

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

"""
    flip_velocity(u, dim)

Flip the `dim`th velocity component of `u`. and return a copy.
"""
function flip_velocity(u::ConservedProps{N,T,U1,U2,U3}, dim) where {N,T,U1,U2,U3}
    scaling = SVector(ntuple(i -> i == dim ? -one(T) : one(T), N))
    return ConservedProps(u.ρ, scaling .* u.ρv, u.ρE)
end