using BenchmarkTools
using Euler2D
using Euler2D: tangent_quadcell_list_and_id_grid, primal_quadcell_list_and_id_grid
using ForwardDiff
using LinearAlgebra
using ShockwaveProperties
using StaticArrays
using Unitful

##

bcs = (
    ExtrapolateToPhantom(), # north 
    ExtrapolateToPhantom(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)
bounds = ((-2.0, 0.0), (-1.5, 1.5))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (50, 75)

starting_parameters = SVector(0.662, 4.0, 0.0, 220.0)

function u0(x, p)
    pp = PrimitiveProps(p[1], SVector(p[2], p[3]), p[4])
    return ConservedProps(pp, DRY_AIR)
end

ambient = u0(nothing, starting_parameters)

x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)
scale = EulerEqnsScaling(x0, ρ0, a0)

global_cells, global_ids = tangent_quadcell_list_and_id_grid(u0, starting_parameters, bounds, ncells, scale, just_circle)

