module Euler2D

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end Euler2D

using Base.Threads: nthreads, @spawn
using LinearAlgebra
using Tullio
using ShockwaveProperties
using ShockwaveProperties: MomentumDensity, EnergyDensity
using StaticArrays
using Unitful
using Unitful: 𝐋, 𝐓, 𝐌, 𝚯, 𝐍
using Unitful: @derived_dimension, Density, Pressure

include("utils.jl")
include("transport.jl")
include("boundary_conditions.jl")
include("riemann_problem.jl")

include("array_simulations/fvm.jl")
include("array_simulations/array_simulations.jl")

include("cell_simulations/obstacle.jl")
include("cell_simulations/cell_simulations.jl")

# methods
export F_euler
export interface_signal_speeds, maximum_Δt
export step_euler_hll!, simulate_euler_equations, simulate_euler_equations_cells

# boundary condition types
export BoundaryCondition, PeriodicAxis, EdgeBoundary
export PhantomEdge
export StrongWall, FixedPhantomOutside, ExtrapolateToPhantom
export SupersonicInflow

# Utility methods
export numeric_dtype

# EulerSim methods
# TODO optimize these to use arrays of SArrays
export EulerSim
export cell_boundaries, cell_centers, nth_step, eachstep
export grid_size, n_data_dims, n_space_dims, n_tsteps
export load_euler_sim

# CellSim methods
export CellBasedEulerSim, RegularQuadCell
export inward_normals, outward_normals, cprops_dtype
export Obstacle, TriangularObstacle, RectangularObstacle, CircularObstacle
export point_inside
export load_cell_sim

function _interpolate_field(fieldfn, sim, t, args...)
    if t < 0 || t > sim.tsteps[end]
        error(
            ArgumentError(
                "Cannot interpolate for values of t outside the range of the simulation.",
            ),
        )
    end
    idx_after = findfirst(>(t), sim.tsteps)
    t1 = sim.tsteps[idx_after-1]
    v1 = fieldfn(sim, idx_after - 1, args...)
    t2 = sim.tsteps[idx_after]
    v2 = fieldfn(sim, idx_after, args...)
    α = (t-t1)/(t2-t1)
    return α*v1 + (1-α)*v2
end

const _field_methods_nogas =
    (:density_field, :velocity_field, :total_internal_energy_density_field)
const _field_methods_gas = (:pressure_field, :mach_number_field)

for T ∈ (EulerSim, CellBasedEulerSim)
    for f ∈ _field_methods_gas
        @eval $(f)(sim::$(T), t, gas::CaloricallyPerfectGas) = begin
            return _interpolate_field($f, sim, t, gas)
        end
    end
    for f ∈ _field_methods_nogas
        @eval $(f)(sim::$(T), t) = begin
            return _interpolate_field($f, sim, t)
        end
    end
end

# All sim methods
export pressure_field, density_field, velocity_field, mach_number_field, total_internal_energy_density_field

end