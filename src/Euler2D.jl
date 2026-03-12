module Euler2D

@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end Euler2D

# standard library
using Dates
using LinearAlgebra
using Statistics

# Load before Tullio
using LoopVectorization

using Accessors: @set, @reset
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using OhMyThreads: tforeach, tmap, tmapreduce, treduce
using StaticArrays
using Tullio
using Unitful
using Unitful: 𝐋, 𝐓, 𝐌, 𝚯, 𝐍
using Unitful: @derived_dimension, Density, Pressure, Velocity

# STCE packages
using PlanePolygons
using ShockwaveProperties
# using ShockwaveProperties: MomentumDensity, EnergyDensity
using SimpleIntegration

const fdiff_backend = AutoForwardDiff()

# reexport
export ShockwaveProperties
export DRY_AIR

# modules
export CannyShockSensor, ShockInterpolation
export ShockSensitivity, LocalPseudoinversion
export FullPseudoinversion

# methods
export F_euler
export interface_signal_speeds
export eigenvectors_∇F_euler, eigenvectors_∇G_euler

# dimension stuff
export EulerEqnsScaling
export nondimensionalize, redimensionalize
export length_scale, mass_scale, time_scale
export density_scale, pressure_scale, velocity_scale
export body_force_scale, energy_density_scale, specific_heat_capacity_scale
export dimensionless_pressure, dimensionless_velocity, dimensionless_ΔS_density
export dimensionless_mach_number, dimensionless_speed_of_sound
export dimensionless_internal_energy, dimensionless_internal_energy_density
export dimensionless_total_enthalpy, dimensionless_total_enthalpy_density
export dimensionless_enthalpy, dimensionless_enthalpy_density

# boundary condition types
export BoundaryCondition, PeriodicAxis, EdgeBoundary
export PhantomEdge
export StrongWall, FixedPhantomOutside, ExtrapolateToPhantom
export SupersonicInflow

# Utility methods
export numeric_dtype

# CellSim methods
export CellBasedEulerSim, FVMCell, PrimalQuadCell, TangentQuadCell
export cell_boundaries, cell_centers, eachstep, nth_step
export grid_size, n_cells, n_space_dims, n_tsteps
export cell_boundary_polygon, cell_center, cell_volume, maximum_cell_size, minimum_cell_size
export Obstacle, TriangularObstacle, RectangularObstacle, CircularObstacle
export load_cell_sim
export cell_simulation_config
export resume_simulation_from_file, start_simulation_from_initial_conditions

# All sim methods
export density_field, pressure_field, velocity_field
export mach_number_field, total_internal_energy_density_field
export estimate_∇u_at, u_and_u̇_at, u_at

include("utils.jl")
include("nondimensionalization.jl")
include("transport.jl")
include("boundary_conditions.jl")

include("riemann_solver/approximate_riemann_solver.jl")
include("riemann_solver/exact_riemann_solver.jl")

include("cell_simulations/cells.jl")
include("cell_simulations/obstacle.jl")
include("cell_simulations/grid.jl")
include("cell_simulations/simulations.jl")
include("cell_simulations/fields.jl")

include("postprocessing/canny_rh_filter.jl")
include("postprocessing/shock_interpolation.jl")
include("postprocessing/shock_sensitivity.jl")
include("postprocessing/local_pseudoinversion.jl")
include("postprocessing/full_pseudoinversion.jl")
include("postprocessing/bressan.jl")

const _SI_DEFAULT_SCALE = EulerEqnsScaling(1.0u"m", 1.0u"kg/m^3", 1.0u"m/s")

function _interpolate_field(fieldfn, sim, t, args...)
    if t < 0 || t > sim.tsteps[end]
        error(
            ArgumentError(
                "Cannot interpolate for values of t outside the range of the simulation.",
            ),
        )
    end
    idx_after = findfirst(>(t), sim.tsteps)
    t1 = sim.tsteps[idx_after - 1]
    t2 = sim.tsteps[idx_after]
    α = (t - t1) / (t2 - t1)
    res = fieldfn(sim, idx_after - 1, args...)
    temp = fieldfn(sim, idx_after, args...)
    for i in eachindex(res, temp)
        if isnothing(res[i])
            continue
        end
        res[i] = α * res[i] + (1 - α) * temp[i]
    end
    return res
end

const _field_methods_nogas =
    (:density_field, :velocity_field, :total_internal_energy_density_field)
const _field_methods_gas = (:pressure_field, :mach_number_field)

for f in _field_methods_gas
    @eval $(f)(sim::CellBasedEulerSim, t, gas::CaloricallyPerfectGas) = begin
        return _interpolate_field($f, sim, t, gas)
    end
    @eval $(f)(
        sim::CellBasedEulerSim,
        t,
        gas::CaloricallyPerfectGas,
        scale::EulerEqnsScaling,
    ) = begin
        return _interpolate_field($f, sim, t, gas, scale)
    end
end
for f in _field_methods_nogas
    @eval $(f)(sim::CellBasedEulerSim, t) = begin
        return _interpolate_field($f, sim, t)
    end
    @eval $(f)(sim::CellBasedEulerSim, t, scale::EulerEqnsScaling) = begin
        return _interpolate_field($f, sim, t, scale)
    end
end

end
