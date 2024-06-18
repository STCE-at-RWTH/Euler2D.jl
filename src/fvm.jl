using LinearAlgebra
using Tullio
using ShockwaveProperties

##

"""
    BoundaryCondition

Abstract supertype for all boundary conditions.
"""
abstract type BoundaryCondition end

"""
    PeriodicAxis

Indicates that this axis should "wrap" back onto itself --
i.e. the right edge of the last cell is the same as the left edge of the first.
"""
struct PeriodicAxis <: BoundaryCondition end

"""
    Edge

Abstract supertype for all edge boundary conditions.
"""
abstract type Edge end

"""
    PhantomEdge{N}

Supertype for all edge conditions that generate a phantom cell 
from `N` neighbors inside the computational domain.
"""
abstract type PhantomEdge{N} <: Edge end

"""
    FluxEdge{N}

Supertype for all edge conditions that generate an edge flux
from `N` cells inside the computational domain.

Currently not used.
"""
abstract type FluxEdge{N} <: Edge end

"""
    reverse_right_edge(::Edge) = true

Check if this edge boundary requires the reversal of velocity components when applied on the right side.
(We assume, by default, that the boundary condition is applied [phantom] - u1 - u2... on the axis.)
"""
reverse_right_edge(::Edge) = true

"""
    nneighbors(::PhantomEdge{N})
    nneighbors(::FluxEdge{N})

How many neighbor cells are required to compute this boundary condition?
"""
nneighbors(::PhantomEdge{N}) where {N} = N
nneighbors(::FluxEdge{N}) where {N} = N

"""
    EdgeBoundary{L, R}

Indicates that this axis has some prescribed left and right edge boundary conditions.
"""
struct EdgeBoundary{L<:Edge,R<:Edge} <: BoundaryCondition
    left::L
    right::R
end

"""
    StrongWall

This edge is a hard wall where the no-penetration condition
is enforced by using a phantom cell.
"""
struct StrongWall <: PhantomEdge{1} end

function phantom_cell(
    ::StrongWall,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    phantom = u
    phantom[1+dim, :] *= -1
    return phantom
end

"""
    FixedPhantomOutside

Fixes a phantom cell outside the given edge of the computational domain.
This BC is appropriate to use if the interesting behavior of the solution
    does **not** approach the boundary.

Fields
---
 - `prescribed_state`: The state prescribed outside the boundary.
"""
struct FixedPhantomOutside <: PhantomEdge{1}
    prescribed_state::ConservedProps
end

function FixedPhantomOutside(s::PrimitiveProps, gas::CaloricallyPerfectGas)
    return FixedPhantomOutside(ConservedProps(s; gas))
end

"""
    reverse_right_edge(::FixedPhantomOutside) = false

This BC does not require velocity vector reversal at the right edge.
"""
reverse_right_edge(::FixedPhantomOutside) = false

function phantom_cell(
    bc::FixedPhantomOutside,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    return state_to_vector(bc.prescribed_state)
end

"""
    ExtrapolateToPhantom

Copies cells to the other side of the domain boundary.
This BC is appropriate to use if the interesting behavior of the solution
    does **not** approach the boundary.
"""
struct ExtrapolateToPhantom <: PhantomEdge{1} end

function phantom_cell(
    bc::ExtrapolateToPhantom,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    return u
end

"""
    SupersonicInflow

Enforces a supersonic inflow condition at the boundary. 

Fields
---
 - `prescribed_state`: the prescribed inflow state. 
"""
struct SupersonicInflow <: PhantomEdge{1}
    prescribed_state::ConservedProps

    function SupersonicInflow(u::ConservedProps, gas::CaloricallyPerfectGas)
        all(>(1.0), mach_number(u; gas)) ||
            ArgumentError("Cannot construct a supersonic inflow boundary with M_∞ ≤ 1.0!")
        return new(u)
    end
end

function SupersonicInflow(s::PrimitiveProps, gas::CaloricallyPerfectGas)
    return SupersonicInflow(ConservedProps(s; gas))
end

function phantom_cell(
    bc::SupersonicInflow,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    return state_to_vector(bc.prescribed_state)
end

"""
    FixedPressureOutflow

Represents an outflow with fixed pressure conditions. It doesn't work.

Fields
---
- `P`: The pressure at the boundary.
"""
struct FixedPressureOutflow <: PhantomEdge{1}
    P
end

"""
    phantom_cell(bc::FixedPressureOutflow, ...)

We take this from the NASA report on Fun3D and fixed-pressure boundary conditions.
Essentially, if the flow is subsonic at the boundary, we fix the pressure 
  and compute density in the phantom.
Otherwise, we assume information gets projected out of the domain.

Note: This will break if the flow becomes supersonic back into the domain. 
That seems reasonable to me.
"""
function phantom_cell(
    bc::FixedPressureOutflow,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    v_i = u[dim+1, 1] / u[1, 1]
    # TODO in 2-D case we will need a boundary normal vector
    M = mach_number(u; gas)[dim]
    # return early if the flow is supersonic
    if abs(M) > 1.0
        return u
    end

    P_i = pressure_u(u[:, 1]; gas)
    ## subsonic flow -> enforce BC, otherwise extrapolate
    P_b = bc.P
    ρ_b = P_b / P_i * u[1, 1]
    # internal energy is c_v⋅T
    T_b = P_b / (u[1, 1] * ustrip(gas.R))
    e_b = T_b / ustrip(gas.c_v)
    ρE_b = ρ_b * (e_b + v_i ⋅ v_i / 2)

    phantom = vcat(ρ_b, ρ_b .* v_i, ρE_b)
    return phantom
end

function left_edge_ϕ(
    bc::PhantomEdge{N},
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T,N}
    phantom = phantom_cell(bc, @view(u[:, 1:N]), dim, gas)
    return ϕ_hll(phantom, @view(u[:, 1]), dim, gas)
end

function right_edge_ϕ(
    bc::PhantomEdge{N},
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T,N}
    neighbors = u[:, end:-1:(end-N+1)] # copy
    # reverse momentum on the right edge
    neighbors[dim+1, :] .*= -1.0
    phantom = phantom_cell(bc, neighbors, dim, gas)
    # reverse the appropriate velocity component
    if reverse_right_edge(bc)
        phantom[1+dim] *= -1
    end
    return ϕ_hll(@view(u[:, end]), phantom, dim, gas)
end

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
    Δt = mapreduce(min, enumerate(zip(dV, boundary_conditions))) do (d, (Δx, bc))
        return mapreduce(min, eachslice(u; dims = d + 1)) do u_ax
            maximum_Δt(u_ax, Δx, bc, d, CFL, gas)
        end
    end
    return Δt
end

# TODO check the signs indside the 2 and 3d versions

"""
    bulk_step!(u_next, u, dV, Δt)

Step the bulk of the simulation grid to the next time step and write the result into `u_next`.
"""
function bulk_step!(
    u_next::AbstractArray{T,2},
    u::AbstractArray{T,2},
    (Δx,),
    Δt,
    gas::CaloricallyPerfectGas,
) where {T}
    @tullio u_next[:, i] = (
        u[:, i] -
        Δt / Δx * (ϕ_hll(u[:, i], u[:, i+1], 1, gas) - ϕ_hll(u[:, i-1], u[:, i], 1, gas))
    )
end

function bulk_step!(
    u_next::AbstractArray{T,3},
    u::AbstractArray{T,3},
    (Δx, Δy),
    Δt,
    gas::CaloricallyPerfectGas,
) where {T}
    @assert length(axes(u, 1)) == 4
    @tullio u_next[:, i, j] = (
        u[:, i, j] - (
            Δt / Δx * (
                ϕ_hll(u[:, i, j], u[:, i+1, j], 1, gas) -
                ϕ_hll(u[:, i-1, j], u[:, i, j], 1, gas)
            ) +
            Δt / Δy * (
                ϕ_hll(u[:, i, j], u[:, i, j+1], 2, gas) -
                ϕ_hll(u[:, i, j-1], u[:, i, j], 2, gas)
            )
        )
    )
end

function bulk_step!(
    u_next::AbstractArray{T,4},
    u::AbstractArray{T,4},
    (Δx, Δy, Δz),
    Δt,
    gas::CaloricallyPerfectGas,
)
    @assert length(axes(u, 1)) == 5
    @tullio u_next[:, i, j] = (
        u[:, i, j, k] - (
            Δt / Δx * (
                ϕ_hll(u[:, i, j, k], u[:, i+1, j, k], 1, gas) -
                ϕ_hll(u[:, i-1, j, k], u[:, i, j, k], 1, gas)
            ) +
            Δt / Δy * (
                ϕ_hll(u[:, i, j, k], u[:, i, j+1, k], 2, gas) -
                ϕ_hll(u[:, i, j-1, k], u[:, i, j, k], 2, gas)
            ) +
            Δt / Δz * (
                ϕ_hll(u[:, i, j, k], u[:, i, j, k+1], 3, gas) -
                ϕ_hll(u[:, i, j, k-1], u[:, i, j, k], 3, gas)
            )
        )
    )
end

""""
    bulk_step_1d_slice!(u_next, u, Δt, Δx, dim, gas)

Apply Godunov's method along a 1-dimensional slice of data ``[ρ, ρv⃗, ρE]``.

The operator splitting method does permit us to handle each real dimension separately.
"""
function bulk_step_1d_slice!(u_next, u, Δt, Δx, dim, gas::CaloricallyPerfectGas)
    @tullio u_next[:, i] = (
        u[:, i] -
        Δt / Δx *
        (ϕ_hll(u[:, i], u[:, i+1], dim, gas) - ϕ_hll(u[:, i-1], u[:, i], dim, gas))
    )
end

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
    u_next[:, 1] = u[:, 1] - (Δt / Δx * (ϕ_hll(u[:, 1], u[:, 2], dim, gas) - ϕ_periodic))
    u_next[:, end] =
        u[:, end] - (Δt / Δx * (ϕ_periodic - ϕ_hll(u[:, end-1], u[:, end], dim, gas)))
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
    u_next[:, 1] =
        u[:, 1] - (
            Δt / Δx * (
                ϕ_hll(u[:, 1], u[:, 2], dim, gas) -
                left_edge_ϕ(boundary_conditions.left, u, dim, gas)
            )
        )
    u_next[:, end] =
        u[:, end] - (
            Δt / Δx * (
                right_edge_ϕ(boundary_conditions.right, u, dim, gas) -
                ϕ_hll(u[:, end-1], u[:, end], dim, gas)
            )
        )
end

function step_euler_hll!(u_next, u, dV, boundary_conditions, Δt, gas::CaloricallyPerfectGas)
    bulk_step!(u_next, u, Δt, dV, gas)
    for (dim, (Δx, bcs)) ∈ enumerate(zip(dV, boundary_conditions))
        enforce_boundary!(u_next, u, bcs, Δx, Δt, gas)
    end
end