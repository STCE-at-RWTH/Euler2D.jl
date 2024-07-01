module Euler2D

using LinearAlgebra
using Tullio
using ShockwaveProperties
using StaticArrays
using Unitful

include("utils.jl")
include("flux1d.jl")
include("boundary_conditions.jl")
include("fvm.jl")
include("array_simulations.jl")
include("cell_simulations.jl")

# boundary condition types
export BoundaryCondition, PeriodicAxis, EdgeBoundary
export PhantomEdge
export StrongWall, FixedPhantomOutside, ExtrapolateToPhantom
export SupersonicInflow

# EulerSim methods
export EulerSim
export cell_boundaries, cell_centers, nth_step, eachstep
export n_data_dims, n_space_dims, n_tsteps
export load_euler_sim

# CellSim methods
export CellBasedEulerSim, RegularQuadCell
export Obstacle, TriangularObstacle, RectangularObstacle, CircularObstacle
export point_inside
export load_cell_sim

# methods
export F_euler
export interface_signal_speeds, maximum_Δt
export step_euler_hll!, simulate_euler_equations, simulate_euler_equations_cells

end