using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ambient = launder_units(PrimitiveProps(0.662, (1.75, 0.0), 220.0))
amb2 = launder_units(PrimitiveProps(0.662, (0.0, 0.0), 220.0))

bc_right = SupersonicInflow(ambient, DRY_AIR)
bc_fix = FixedPhantomOutside(ambient)
bcs = (
    bc_fix, # north 
    bc_fix, # south
    ExtrapolateToPhantom(), # east
    bc_fix, # west
    StrongWall(), # walls
)
bounds = ((-4.0, 0.0), (-4.0, 4.0))
just_circle = [CircularObstacle((0.0, 0.0), 0.75)]
just_triangle = [TriangularObstacle((-0.5, 0.0), (0.25, 0.5), (0.25, -0.5))]
ncells = (400,800)

sim_results = Euler2D.simulate_euler_equations_cells(
    0.1,
    bcs,
    just_circle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 1,
    max_tsteps = 10,
    output_tag = "circular_obstacle_radius_1",
    tasks_per_axis = 4,
    output_channel_size = 50,
) do (x, y)
    ambient
end;

Euler2D.simulate_euler_equations_cells(
    0.1,
    bcs,
    just_circle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 25,
    max_tsteps = 10,
    output_tag = "circular_obstacle_radius_1_compound_shock",
) do (x, y)
    y > 0.0 ? amb2 : ambient
end

Euler2D.simulate_euler_equations_cells(
    0.1,
    bcs,
    just_triangle,
    bounds,
    ncells;
    gas = DRY_AIR,
    info_frequency = 25,
    max_tsteps = 10,
    output_tag = "funky_triangle",
) do (x, y)
    ambient
end