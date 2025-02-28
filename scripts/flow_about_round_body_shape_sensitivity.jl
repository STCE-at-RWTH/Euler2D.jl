using Euler2D
using LinearAlgebra
using Unitful
using ShockwaveProperties
using StaticArrays
using Printf

function u0(x, p)
    ρ∞  = p[1]
    M∞  = p[2]
    T∞  = p[3]

    pp = PrimitiveProps(ρ∞, SVector(M∞, 0.0), T∞)
    return ConservedProps(pp, DRY_AIR)
end


function create_circle(p)
    center = (0.0, 0.0)
    radius = p[4]
    return CircularObstacle(center, radius)
end

bcs = (
    ExtrapolateToPhantom(), # north 
    ExtrapolateToPhantom(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)
bounds = ((-2.0, 0.0), (-1.5, 1.5))
ncells = (50, 75)
tsteps = 10

starting_parameters = SVector(0.662, 4.0, 220.0, 0.75)

ambient = u0(nothing, starting_parameters)

x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)
scale = EulerEqnsScaling(x0, ρ0, a0)

obstacle = create_circle(starting_parameters)
just_circle = [obstacle]

Euler2D.simulate_euler_equations_cells(
    u0,
    starting_parameters,
    1.0,
    bcs,
    just_circle,
    bounds,
    ncells;
    mode = Euler2D.TANGENT,
    gas = DRY_AIR,
    scale = scale,
    info_frequency = 1,
    write_frequency = 1,
    max_tsteps = tsteps,
    output_tag = "circular_obstacle_tangent",
    output_channel_size = 2,
    tasks_per_axis = 2,
);  

tangent=load_cell_sim("data/circular_obstacle_tangent.celltape");

cells = tangent.cells[tsteps]
println("Sensitivity of radius to flow properties at final time step")
for (cell_id, cell) in cells
    if cell.contains_boundary
        radius_sensitivity = cell.u̇[:, 4]
        formatted_rs = join(map(x -> @sprintf("%.4g", x), radius_sensitivity), ", ")
        println("Cell ", cell_id, " at ", cell.center, ": [", formatted_rs, "]")
    end
end
