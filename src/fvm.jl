using LinearAlgebra
using Tullio
using ShockwaveProperties

##

abstract type BoundaryCondition end
struct PeriodicAxis <: BoundaryCondition end


abstract type SolidWall end
struct StrongWallReflect <: SolidWall end
struct WeakWallReflect <: BoundaryCondition end
struct WeakWallExtrapolate <: BoundaryCondition end

struct WallBoundary{LEFT<:SolidWall,RIGHT<:SolidWall} <: BoundaryCondition end

struct Inflow <: BoundaryCondition end
struct Outflow <: BoundaryCondition end

##

"""
Computes the numerical flux when the boundary of a cell is a wall.

We know that "weakly" enforcing the boundary condition sets the flux at the wall to `[0 0; P 0; 0 P; 0 0]`.
"""
function ϕ_wall(::WeakWallReflect, u, dim; gas::CaloricallyPerfectGas)
    return pressure_u(u; gas = gas) * I[1:length(u), dim+1]
end

"""
Extrapolates the pressure at the wall by using the neighbor cell on the other side.
"""
function ϕ_extrapolate_wall(u1, u2, dim; gas::CaloricallyPerfectGas)
    return (1.5 * pressure_u(u1; gas = gas) - 0.5 * pressure_u(u2; gas = gas)) *
           I[1:length(u), dim+1]
end

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
    gas::CaloricallyPerfectGas = DRY_AIR,
) where {U<:AbstractArray{Float64,2}}
    @assert size(u)[1] == 3
    @tullio u_next[:, i] =
        u[:, i] + (
            Δt / Δx * (
                ϕ_hll(u[:, i-1], u[:, i], 1; gas = gas) -
                ϕ_hll(u[:, i], u[:, i+1], 1; gas = gas)
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
    Δt::Float64,
    Δx::Float64,
    Δy::Float64;
    gas::CaloricallyPerfectGas = DRY_AIR,
) where {U<:AbstractArray{Float64,3}}
    @assert size(u)[1] == 4
    @tullio u_next[:, i, j] =
        u[:, i, j] + (
            Δt / Δx * (
                ϕ_hll(u[:, i-1, j], u[:, i, j], 1; gas = gas) -
                ϕ_hll(u[:, i, j], u[:, i+1, j], 1; gas = gas)
            ) +
            Δt / Δy * (
                ϕ_hll(u[:, i, j-1], u[:, i, j], 2; gas = gas) -
                ϕ_hll(u[:, i, j], u[:, i, j+1], 2; gas = gas)
            )
        )
end

##

function enforce_boundary!(
    ::WallBoundary{LEFT,RIGHT},
    u_next::U,
    u::U,
    Δx::Float64,
    Δt::Float64;
    gas::CaloricallyPerfectGas,
) where {U<:AbstractArray{Float64,2},LEFT,RIGHT}
    phantom = [u[1, 1], -1 * u[2, 1], u[3, 1]]
    u_next[:, 1] =
        u[:, 1] + (
            Δt / Δx *
            (ϕ_hll(phantom, u[:, 1], 1; gas = gas) - ϕ_hll(u[:, 1], u[:, 2], 1; gas = gas))
        )
    phantom = [u[1, end], -1 * u[2, end], u[3, end]]
    u_next[:, end] =
        u[:, end] + (
            Δt / Δx * (
                ϕ_hll(u[:, end-1], u[:, end], 1; gas = gas) -
                ϕ_hll(u[:, end], phantom, 1; gas = gas)
            )
        )
end

function enforce_boundary!(
    ::PeriodicAxis,
    u_next::U,
    u::U,
    Δx,
    Δt;
    gas::CaloricallyPerfectGas,
) where {U<:AbstractArray{Float64,2}}
    u_next[:, 1] =
        u[:, 1] + (
            Δt / Δx * (
                ϕ_hll(u[:, end], u[:, 1], 1; gas = gas) -
                ϕ_hll(u[:, 1], u[:, 2], 1; gas = gas)
            )
        )
    u_next[:, end] =
        u[:, end] + (
            Δt / Δx * (
                ϕ_hll(u[:, end-1], u[:, end], 1; gas = gas) -
                ϕ_hll(u[:, end], u[:, 1], 1; gas = gas)
            )
        )
end

##

"""
    maximum_Δt(<:BoundaryCondition, u, Δx, CFL, dim; gas)

Compute the maximum possible `Δt` given a `Δx` and CFL number.
"""
function maximum_Δt(::PeriodicAxis, u, Δx, CFL, dim; gas::CaloricallyPerfectGas)
    a = mapreduce(max, zip(eachcol(@view(u[:, 1:end-1])), eachcol(@view(u[:, 2:end])))) do (uL, uR)
        max(interface_signal_speeds(uL, uR, dim; gas=gas)...)
    end
    a_bc = max(interface_signal_speeds(u[:, end], u[:, 1], dim; gas=gas)...)
    a = max(a, a_bc)
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
) where {U<:AbstractArray{Float64, 2}}
    bulk_step!(u_next, u, Δt, Δx; gas = gas)
    enforce_boundary!(x_bcs, u_next, u, Δt, Δx; gas = gas)
end

##

# (a, b) = simulate_euler_2d(100.0, 100.0, 50, 50, 1.0, u0; max_tsteps = 10000)