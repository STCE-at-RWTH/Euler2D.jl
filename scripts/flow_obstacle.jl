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
just_triangle = [TriangularObstacle((-0.75, 0.0), (1.0, 1.0), (1.0, -1.0))]
just_square = [RectangularObstacle(SVector(0., 0.), SVector(0.5, 0.5))]
ncells = (300,450)

##

Euler2D.simulate_euler_equations_cells(
    1.0,
    bcs,
    just_circle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 25,
    write_frequency = 100,
    max_tsteps = 25000,
    output_tag = "circular_obstacle_longtime",
    output_channel_size = 2,
    tasks_per_axis = 4,
) do (x, y)
    ambient
end;