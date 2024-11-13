using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (4.0, 0.0), 220.0))
amb2 = launder_units(PrimitiveProps(0.662, (0.0, 0.0), 220.0))

bc_right = SupersonicInflow(ambient, DRY_AIR)
bc_fix = FixedPhantomOutside(ambient)
bcs = (
    ExtrapolateToPhantom(), # north 
    ExtrapolateToPhantom(), # south
    bc_right, # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)
bounds = ((-2.0, 0.0), (-1.5, 1.5))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
ncells = (50,75)

function u0(x, y, params)
    pp = PrimitiveProps(params[1], (params[2], params[3]), params[4])
    return launder_units(pp)
end

function seeds(x, y, params)

end

## TODO try nondimensionalizing? we don't really seem to be gaining any benefit and AD is much easier
## when non-dimensionalized
## Then we can just work with SVectors rather than ConservedProps... even though we'd really like to keep
## the security that dimensionful quantities offer, this really isn't working out.
## shouldn't be too hard to fix in the flux funciton...
## https://en.wikipedia.org/wiki/Cauchy_momentum_equation#Nondimensionalisation 

##

# Euler2D.simulate_euler_equations_cells(
#     0.25,
#     bcs,
#     just_circle,
#     bounds,
#     ncells;
#     gas = DRY_AIR,
#     info_frequency = 25,
#     write_frequency = 100,
#     max_tsteps = 25000,
#     output_tag = "circular_obstacle_with_tangents",
#     output_channel_size = 2,
#     tasks_per_axis = 4,
# ) do (x, y)
#     ambient
# end;