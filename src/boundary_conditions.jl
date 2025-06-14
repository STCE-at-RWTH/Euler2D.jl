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
    # HACK this isn't good.
    phantom = reshape(u, length(u))
    phantom[1+dim] *= -1
    return phantom
end

function phantom_cell(::StrongWall, u, dim, gas::CaloricallyPerfectGas)
    return flip_velocity(u, dim)
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

function phantom_cell(bc::FixedPhantomOutside, u::ConservedProps, dim, gas)
    return bc.prescribed_state
end

"""
    ExtrapolateToPhantom

Copies cells to the other side of the domain boundary.
This BC is appropriate to use if the interesting behavior of the solution
    does **not** approach the boundary.
"""
struct ExtrapolateToPhantom <: PhantomEdge{1} end

function phantom_cell(bc::ExtrapolateToPhantom, u, dim, gas::CaloricallyPerfectGas)
    return reshape(u, length(u))
end

function phantom_cell(
    ::ExtrapolateToPhantom,
    u::ConservedProps,
    dim,
    gas::CaloricallyPerfectGas,
)
    #TODO update ShockwaveProperties to have copy and parametrized methods for construction
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
        all(>(1.0), mach_number(u, gas)) ||
            ArgumentError("Cannot construct a supersonic inflow boundary with M_∞ ≤ 1.0!")
        return new(u)
    end
end

function SupersonicInflow(s::PrimitiveProps, gas::CaloricallyPerfectGas)
    return SupersonicInflow(ConservedProps(s, gas))
end

function phantom_cell(
    bc::SupersonicInflow,
    u::AbstractArray{T,2},
    dim,
    gas::CaloricallyPerfectGas,
) where {T}
    return state_to_vector(bc.prescribed_state)
end

function phantom_cell(
    bc::SupersonicInflow,
    u::ConservedProps,
    dim,
    gas::CaloricallyPerfectGas,
)
    return bc.prescribed_state
end
