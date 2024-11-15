using Euler2D
using LinearAlgebra
using Unitful
using ShockwaveProperties
using StaticArrays

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (4.0, 0.0), 220.0))

x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)

scale = EulerEqnsScaling(x0, ρ0, a0)

bc_right = SupersonicInflow(ambient, DRY_AIR)
bc_fix = FixedPhantomOutside(ambient)
bcs = (
    ExtrapolateToPhantom(), # north 
    ExtrapolateToPhantom(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)
bounds = ((-2.0, 0.0), (-1.5, 1.5))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
just_triangle = [TriangularObstacle((-0.75, 0.0), (1.0, 1.0), (1.0, -1.0))]
just_square = [RectangularObstacle(SVector(0.0, 0.0), SVector(0.5, 0.5))]
ncells = (50, 75)

##

Euler2D.simulate_euler_equations_cells(
    1.0,
    bcs,
    just_circle,
    bounds,
    ncells;
    gas = DRY_AIR,
    scale = scale,
    info_frequency = 5,
    write_frequency = 5,
    max_tsteps = 1000,
    output_tag = "circular_obstacle_radius_1",
    output_channel_size = 2,
    tasks_per_axis = 2,
) do (x, y)
    ambient
end;

# Euler2D.simulate_euler_equations_cells(
#     1.0,
#     bcs,
#     just_circle,
#     bounds,
#     ncells;
#     gas = DRY_AIR,
#     info_frequency = 25,
#     write_frequency = 100,
#     max_tsteps = 25000,
#     output_tag = "circular_obstacle_longtime",
#     output_channel_size = 2,
#     tasks_per_axis = 4,
# ) do (x, y)
#     ambient
# end;