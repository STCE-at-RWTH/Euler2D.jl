module Euler2D

using LinearAlgebra
using Tullio
using ShockwaveProperties
using Unitful

include("flux1d.jl")
include("flux2d.jl")
include("fvm.jl")

export BoundaryCondition, PeriodicAxis, WallBoundary, StrongWall, WeakWallReflect, WeakWallExtrapolate
export interface_signal_speeds, maximum_Δt
export step_euler_hll!, bulk_step!, enforce_boundary!

end