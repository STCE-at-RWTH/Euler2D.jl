module Euler2D

using LinearAlgebra
using LoopVectorization
using Tullio
using ShockwaveProperties
using Unitful

include("flux1d.jl")
include("flux2d.jl")
include("fvm.jl")

# boundary condition types
export BoundaryCondition, PeriodicAxis, EdgeBoundary
export PhantomEdge
export StrongWall, FixedPhantomOutside, ExtrapolateToPhantom
export SupersonicInflow, FixedPressureOutflow

# methods
export interface_signal_speeds, maximum_Δt
export step_euler_hll!, bulk_step!, enforce_boundary!

end