using LinearAlgebra
using Tullio
using ShockwaveProperties

##

"""
    maximum_Δt(u, Δx, boundary_condition, dim, cfl_limit, gas)

Compute the maximum possible `Δt` given a `Δx` and CFL upper bound.
"""
function maximum_Δt(u, Δx, ::PeriodicAxis, dim, cfl_limit, gas::CaloricallyPerfectGas)
    a = mapreduce(
        max,
        zip(eachcol(@view(u[:, 1:end-1])), eachcol(@view(u[:, 2:end]))),
    ) do (uL, uR)
        max(abs.(interface_signal_speeds(uL, uR, dim, gas))...)
    end
    a_bc = max(abs.(interface_signal_speeds(u[:, end], u[:, 1], dim, gas))...)
    a = max(a, a_bc)
    Δt = cfl_limit * Δx / a
    return Δt
end

function maximum_Δt(
    u,
    Δx,
    bcs::EdgeBoundary{L,R},
    dim,
    cfl_limit,
    gas::CaloricallyPerfectGas,
) where {L<:PhantomEdge,R<:PhantomEdge}
    a = mapreduce(
        max,
        zip(eachcol(@view(u[:, 1:end-1])), eachcol(@view(u[:, 2:end]))),
    ) do (uL, uR)
        max(abs.(interface_signal_speeds(uL, uR, dim, gas))...)
    end

    phantom_L = phantom_cell(bcs.left, u[:, 1:nneighbors(bcs.left)], dim, gas)
    neighbors_R = u[:, end:-1:(end-nneighbors(bcs.right)+1)]
    neighbors_R[2, :] *= -1
    phantom_R = phantom_cell(bcs.right, neighbors_R, dim, gas)
    if reverse_right_edge(bcs.right)
        phantom_R[2] *= -1
    end
    a = max(a, abs.(interface_signal_speeds(phantom_L, u[:, 1], dim, gas))...)
    a = max(a, abs.(interface_signal_speeds(u[:, end], phantom_R, dim, gas))...)
    Δt = cfl_limit * Δx / a
    return Δt
end

"""
    maximum_Δt(u, dV, boundary_conditions, cfl_lmit, gas)

Compute the maximum possible `Δt` for the data `u` with cell spacing `dV`, 
    given `boundary_conditions` and a `cfl_limit`. 
"""
function maximum_Δt(u, dV, boundary_conditions, CFL, gas::CaloricallyPerfectGas)
    Δt = mapreduce(min, enumerate(zip(dV, boundary_conditions))) do (space_dim, (Δx, bc))
        dims = ((i + 1 for i ∈ 1:length(dV) if i ≠ space_dim)...,)
        u_ax = eachslice(u; dims)
        Δts = Vector{eltype(u)}(undef, length(u_ax))
        Threads.@threads for i ∈ eachindex(u_ax)
            Δts[i] = maximum_Δt(u_ax[i], Δx, bc, space_dim, CFL, gas)
        end
        minimum(Δts)
    end
    return Δt
end

"""
    bulk_step_1d_slice!(u_next, u, Δt, Δx, dim, gas)

Apply Godunov's method along a 1-dimensional slice of data ``[ρ, ρv⃗, ρE]``.

The operator splitting method does permit us to handle each real dimension separately.
"""
function bulk_step_1d_slice!(u_next, u, Δt, Δx, dim, gas::CaloricallyPerfectGas)
    # this expression is different from enforce_boundary because
    #   tullio cannot parse -=
    @tullio u_next[:, i] += (
        Δt / Δx *
        (ϕ_hll(u[:, i-1], u[:, i], dim, gas) - ϕ_hll(u[:, i], u[:, i+1], dim, gas))
    )
end

"""
    enforce_boundary_1d_slice!(u_next, u, Δt, Δx, boundary_conditions, dim, gas)

Enforce a boundary condition on a 1-d slice of data ``[ρ, ρv⃗, ρE]``.

## Arguments
- `u_next`: Output array of data for the results at the next time step
- `u`: Input data at the current time step
- `Δt`: Time step size
- `Δx`: The inter-cell spacing on real axis `dim`
- `boundary_conditions`: The boundary condition to enforce
- `dim`: The real axis being sliced
- `gas`
"""
function enforce_boundary_1d_slice!(
    u_next::AbstractArray{T,2},
    u::AbstractArray{T,2},
    Δt,
    Δx,
    boundary_conditions::PeriodicAxis,
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    # flux out of the last cell into the first cell
    ϕ_periodic = ϕ_hll(u[:, end], u[:, 1], dim, gas)
    u_next[:, 1] -= (Δt / Δx * (ϕ_hll(u[:, 1], u[:, 2], dim, gas) - ϕ_periodic))
    u_next[:, end] -= (Δt / Δx * (ϕ_periodic - ϕ_hll(u[:, end-1], u[:, end], dim, gas)))
end

function enforce_boundary_1d_slice!(
    u_next::AbstractArray{T,2},
    u::AbstractArray{T,2},
    Δt,
    Δx,
    boundary_conditions::EdgeBoundary,
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    u_next[:, 1] -= (
        Δt / Δx * (
            ϕ_hll(u[:, 1], u[:, 2], dim, gas) -
            left_edge_ϕ(boundary_conditions.left, u, dim, gas)
        )
    )
    u_next[:, end] -= (
        Δt / Δx * (
            right_edge_ϕ(boundary_conditions.right, u, dim, gas) -
            ϕ_hll(u[:, end-1], u[:, end], dim, gas)
        )
    )
end

"""
    step_euler_hll!(u_next, u, Δt, dV, bcs, gas)

Step the Euler equations one `Δt` into the future, and write the result into `u_next`.
"""
function step_euler_hll!(u_next, u, Δt, dV, boundary_conditions, gas::CaloricallyPerfectGas)
    @assert length(dV) == length(boundary_conditions)
    @assert size(u_next) == size(u)
    N = length(dV)
    # first part of the update step
    copyto!(u_next, u)
    for (space_dim, (Δx, bcs)) ∈ enumerate(zip(dV, boundary_conditions))
        # there must be a better way
        dims = ((i + 1 for i ∈ 1:N if i ≠ space_dim)...,)
        u_slices = collect(zip(eachslice(u_next; dims), eachslice(u; dims)))
        Threads.@threads for i in eachindex(u_slices)
            u_next_slice, u_slice = u_slices[i]
            # compute flux difference
            bulk_step_1d_slice!(u_next_slice, u_slice, Δt, Δx, space_dim, gas)
            # and on the boundary...
            enforce_boundary_1d_slice!(u_next_slice, u_slice, Δt, Δx, bcs, space_dim, gas)
        end
    end
end