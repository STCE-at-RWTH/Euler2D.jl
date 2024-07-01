n_space_dims(::ConservedProps{N,T,U1,U2,U3}) where {N,T,U1,U2,U3} = N

u_array_space_dims(::AbstractArray{T,N}) where {T,N} = N - 1
u_array_space_size(u::AbstractArray{T,N}) where {T,N} = size(u)[2:end]

function free_space_dims(N, d)
    ((i + 1 for i ∈ 1:N if i ≠ d)...,)
end

function boundary_velocity_scaling(T, dim, rev)
    return SVector(ntuple(4) do i
        (i == dim + 1) && rev ? -one(T) : one(T)
    end)
end
