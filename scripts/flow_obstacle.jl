using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

# the Unitful package likes to come up with wacky units that are dimensionally equivalent to standard metric ones
# this just "launders" these creative ones back to metric base units
# TODO make sure promoting unitful state types between each other works out.
function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (4.0, 0.0), 220.0))

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

# adjust this parameter if you want to change the resolution
#   75 x 150 also gives reasonable results
#   400 x 600 generates a real good picture
ncells = (200,300)

##

Euler2D.simulate_euler_equations_cells(
    0.25,
    bcs,
    just_circle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 10,
    write_frequency = 5,
    max_tsteps = 1000,
    output_tag = "circular_obstacle_radius_1",
    tasks_per_axis = 2,
) do (x, y)
    ambient
end;