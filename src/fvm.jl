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
    EdgeBoundary{L, R}

Indicates that this axis has some prescribed left and right edge boundary conditions.
"""
struct EdgeBoundary{L<:Edge,R<:Edge} <: BoundaryCondition
    left::L
    right::R
end

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
"""
abstract type FluxEdge{N} end

"""
    reverse_right_edge(::Edge) = true

Check if this edge boundary requires the reversal of velocity components when applied on the right side.
(We assume, by default, that the boundary condition is applied [phantom] - 1 - 2... on the axis.)
"""
reverse_right_edge(::Edge) = true

"""
    StrongWall

This edge is a hard wall where the no-penetration condition
is enforced by using a phantom cell.
"""
struct StrongWall <: PhantomEdge{1} end

function phantom_cell(
    ::StrongWall,
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T}
    phantom = u
    phantom[1+dim, :] *= -1
    return phantom
end

struct WeakWallReflect <: FluxEdge{1} end
# struct WeakWallExtrapolate <: FluxEdge{2} end

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
    dim;
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
    dim;
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
            error("Cannot construct a supersonic inflow boundary with M_∞ ≤ 1.0!")
        return new(u)
    end
end

function SupersonicInflow(s::PrimitiveProps, gas::CaloricallyPerfectGas)
    return SupersonicInflow(ConservedProps(s; gas))
end

function phantom_cell(
    bc::SupersonicInflow,
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T}
    return state_to_vector(bc.prescribed_state)
end

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
    dim;
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

# struct FixedMassFlow end

##

"""
    bulk_step!(u_next, u, Δt, Δx; gas)

Step the bulk of the simulation grid to the next time step and write the result into `u_next`.
- `u_next` and `u` are `3 x Nx` grids.
"""
function bulk_step!(
    u_next::U,
    u::U,
    Δt::Float64,
    Δx::Float64;
    gas::CaloricallyPerfectGas,
) where {U<:AbstractArray{Float64,2}}
    @assert size(u)[1] == 3
    @tullio u_next[:, i] = (
        u[:, i] - (
            Δt / Δx *
            (ϕ_hll(u[:, i], u[:, i+1], 1; gas) - ϕ_hll(u[:, i-1], u[:, i], 1; gas))
        )
    )
end

"""
    bulk_step!(u_next, u, Δt, Δx, Δy; gas)

Step the bulk of the simulation grid to the next time step and write the result into `u_next`.
- `u_next` and `u` are `4 × Nx × Ny`, e.g. `4x100x100` for a 10,000 grid cell simulation.
"""
function bulk_step!(
    u_next::U,
    u::U,
    Δt,
    Δx,
    Δy;
    gas::CaloricallyPerfectGas,
) where {U<:AbstractArray{Float64,3}}
    @assert size(u)[1] == 4
    @tullio u_next[:, i, j] = (
        u[:, i, j] - (
            Δt / Δx * (
                ϕ_hll(u[:, i, j], u[:, i+1, j], 1; gas) -
                ϕ_hll(u[:, i-1, j], u[:, i, j], 1; gas)
            ) -
            Δt / Δy * (
                ϕ_hll(u[:, i, j], u[:, i, j+1], 2; gas) -
                ϕ_hll(u[:, i, j-1], u[:, i, j], 2; gas)
            )
        )
    )
end

##

##
## NOTES FROM WEDNESDAY
## I think there's a sign error somewhere in this boundary correction.
## we get reasonable answers if I flip the sign in the equation to find T from rhoE
## when rhoE is negative
## i'm so confused.
## 

function enforce_boundary!(
    ::PeriodicAxis,
    u_next::AbstractArray{T,2},
    u::AbstractArray{T,2},
    Δx::T,
    Δt::T;
    gas::CaloricallyPerfectGas,
) where {T}
    # flux out of the last cell into the first cell
    ϕ_periodic = ϕ_hll(u[:, end], u[:, 1], 1; gas)
    u_next[:, 1] = u[:, 1] - (Δt / Δx * (ϕ_hll(u[:, 1], u[:, 2], 1; gas) - ϕ_periodic))
    u_next[:, end] =
        u[:, end] - (Δt / Δx * (ϕ_periodic - ϕ_hll(u[:, end-1], u[:, end], 1; gas)))
end

function enforce_boundary!(
    bcs::EdgeBoundary{L,R},
    u_next::AbstractArray{T,2},
    u::AbstractArray{T,2},
    Δx,
    Δt;
    gas::CaloricallyPerfectGas,
) where {L,R,T}
    u_next[:, 1] =
        u[:, 1] -
        (Δt / Δx * (ϕ_hll(u[:, 1], u[:, 2], 1; gas) - left_edge_ϕ(bcs.left, u, 1; gas)))

    u_next[:, end] =
        u[:, end] - (
            Δt / Δx *
            (right_edge_ϕ(bcs.right, u, 1; gas) - ϕ_hll(u[:, end-1], u[:, end], 1; gas))
        )
end

# TODO we actually need to flip the the cell momenta passed into the flux calculation
# TODO   and then flip the result (treat everything as the LEFT edge inside edge_flux)

function left_edge_ϕ(
    bc::PhantomEdge{N},
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T,N}
    phantom = phantom_cell(bc, @view(u[:, 1:N]), dim; gas)
    return ϕ_hll(phantom, @view(u[:, 1]), dim; gas)
end

function right_edge_ϕ(
    bc::PhantomEdge{N},
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T,N}
    neighbors = u[:, end:-1:(end-N+1)] # copy
    # reverse momentum on the right edge
    neighbors[dim+1, :] .*= -1.0
    phantom = phantom_cell(bc, neighbors, dim; gas)
    # reverse the appropriate velocity component
    if reverse_right_edge(bc)
        phantom[1+dim] *= -1
    end
    return ϕ_hll(@view(u[:, end]), phantom, dim; gas)
end

function left_edge_ϕ(
    bc::FluxEdge{N},
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T,N}
    return edge_flux(bc, @view(u[:, 1:N]), dim; gas)
end

# TODO we actually need to flip the the cell momenta passed into the flux calculation
# TODO   and then flip the result (treat everything as the LEFT edge inside edge_flux)

function right_edge_ϕ(
    bc::FluxEdge{N},
    u::AbstractArray{T,2},
    dim;
    gas::CaloricallyPerfectGas,
) where {T,N}
    neighbors = u[:, end:-1:(end-N+1)] # copy
    neighbors[dim+1, :] .*= -1.0
    return edge_flux(bc, neighbors, dim; gas)
end

##

"""
    maximum_Δt(<:BoundaryCondition, u, Δx, CFL, dim; gas)

Compute the maximum possible `Δt` given a `Δx` and CFL number.
"""
function maximum_Δt(::PeriodicAxis, u, Δx, CFL, dim; gas::CaloricallyPerfectGas)
    a = mapreduce(
        max,
        zip(eachcol(@view(u[:, 1:end-1])), eachcol(@view(u[:, 2:end]))),
    ) do (uL, uR)
        max(abs.(interface_signal_speeds(uL, uR, dim; gas))...)
    end
    a_bc = max(abs.(interface_signal_speeds(u[:, end], u[:, 1], dim; gas))...)
    a = max(a, a_bc)
    Δt = CFL * Δx / a
    return Δt
end

function maximum_Δt(::EdgeBoundary, u, Δx, CFL, dim; gas::CaloricallyPerfectGas)
    a = mapreduce(
        max,
        zip(eachcol(@view(u[:, 1:end-1])), eachcol(@view(u[:, 2:end]))),
    ) do (uL, uR)
        max(abs.(interface_signal_speeds(uL, uR, dim; gas))...)
    end
    Δt = CFL * Δx / a
    return Δt
end

function step_euler_hll!(
    u_next::U,
    u::U,
    Δt,
    Δx,
    x_bcs::BoundaryCondition;
    gas::CaloricallyPerfectGas,
) where {U<:AbstractArray{Float64,2}}
    bulk_step!(u_next, u, Δt, Δx; gas)
    enforce_boundary!(x_bcs, u_next, u, Δt, Δx; gas)
end

##

# (a, b) = simulate_euler_2d(100.0, 100.0, 50, 50, 1.0, u0; max_tsteps = 10000)