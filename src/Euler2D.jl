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
using ShockwaveProperties: MomentumDensity, EnergyDensity
using SimpleIntegration

const fdiff_backend = AutoForwardDiff()

# reexport
export ShockwaveProperties
export DRY_AIR

# methods
export F_euler
export interface_signal_speeds, maximum_Δt
export eigenvectors_∇F_euler, eigenvectors_∇G_euler
export simulate_euler_equations_cells

# dimension stuff
export EulerEqnsScaling
export nondimensionalize, redimensionalize
export length_scale, time_scale, mass_scale
export velocity_scale, density_scale, pressure_scale
export energy_density_scale, body_force_scale, specific_heat_capacity_scale
export dimensionless_velocity, dimensionless_pressure, dimensionless_ΔS_density
export dimensionless_mach_number, dimensionless_speed_of_sound
export dimensionless_internal_energy_density, dimensionless_internal_energy
export dimensionless_total_enthalpy_density, dimensionless_total_enthalpy
export dimensionless_enthalpy_density, dimensionless_enthalpy
export shift_velocity_coordinates

# boundary condition types
export BoundaryCondition, PeriodicAxis, EdgeBoundary
export PhantomEdge
export StrongWall, FixedPhantomOutside, ExtrapolateToPhantom
export SupersonicInflow

# Utility methods
export numeric_dtype

# CellSim methods
export CellBasedEulerSim, FVMCell, PrimalQuadCell, TangentQuadCell
export inward_normals, outward_normals
export cell_boundaries, cell_centers, nth_step, eachstep
export grid_size, n_data_dims, n_space_dims, n_tsteps
export cell_volume, cell_boundary_polygon, minimum_cell_size, maximum_cell_size
export Obstacle, TriangularObstacle, RectangularObstacle, CircularObstacle
export load_cell_sim

# All sim methods
export pressure_field, density_field, velocity_field
export mach_number_field, total_internal_energy_density_field
export ∇u_at

include("utils.jl")
include("nondimensionalization.jl")
include("transport.jl")
include("boundary_conditions.jl")
include("riemann_solver/approximate_riemann_solver.jl")
include("riemann_solver/exact_riemann_solver.jl")
include("array_simulations/fvm.jl")
include("array_simulations/array_simulations.jl")
include("cell_simulations/cells.jl")
include("cell_simulations/obstacle.jl")
include("cell_simulations/grid.jl")
include("cell_simulations/simulations.jl")
include("postprocessing/canny_rh_filter.jl")
include("postprocessing/local_pseudoinversion.jl")

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
    t1 = sim.tsteps[idx_after-1]
    t2 = sim.tsteps[idx_after]
    α = (t - t1) / (t2 - t1)
    res = fieldfn(sim, idx_after - 1, args...)
    temp = fieldfn(sim, idx_after, args...)
    for i ∈ eachindex(res, temp)
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

for f ∈ _field_methods_gas
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
for f ∈ _field_methods_nogas
    @eval $(f)(sim::CellBasedEulerSim, t) = begin
        return _interpolate_field($f, sim, t)
    end
    @eval $(f)(sim::CellBasedEulerSim, t, scale::EulerEqnsScaling) = begin
        return _interpolate_field($f, sim, t, scale)
    end
end

end
