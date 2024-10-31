module Euler2D

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end Euler2D

# what do we want out of Base?
using Base.Threads: nthreads, @spawn

using Accessors
using ForwardDiff
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
include("cell_simulations/grid.jl")
include("cell_simulations/simulations.jl")

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
export CellBasedEulerSim, PrimalQuadCell
export inward_normals, outward_normals, cprops_dtype
export Obstacle, TriangularObstacle, RectangularObstacle, CircularObstacle
export point_inside
export load_cell_sim

# All sim methods
export pressure_field, density_field, velocity_field, mach_number_field

end