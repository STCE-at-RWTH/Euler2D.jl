using LinearAlgebra
using Unitful
using StaticArrays

using Euler2D
using ShockwaveProperties

"""
    u0(x, p)

Accepts ``x∈ℝ^2`` and a vector of three parameters: 
 - free stream density
 - mach number, 
 - and temperature (understood in metric base units)
"""
function u0(x, p)
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

starting_parameters = SVector(0.662, 4.0, 220.0)
ambient = u0(nothing, starting_parameters)

# set the nondimensionalization scale
x0 = 1.0u"m"
a0 = speed_of_sound(ambient, DRY_AIR)
ρ0 = density(ambient)
scale = EulerEqnsScaling(x0, ρ0, a0)

@info "Current nondimensionalization scale is:" x_0 = length_scale(scale) v_0 =
    velocity_scale(scale) ρ_0 = density_scale(scale)

const bcs = (
    ExtrapolateToPhantom(), # north 
    StrongWall(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)

const bounds = ((-1.5, 0.0), (0.0, 2.0))
probe = [
    CircularObstacle((0.0, 0.0), 0.75),
    RectangularObstacle(SVector(1.0, 0.0), SVector(2.0, 1.5)),
]
const ncells = (240, 240 * 4 ÷ 3)

# Profile.init(; n = 10^7, delay = 0.005)
# Profile.clear()

sim_config = Euler2D.cell_simulation_config(;
    mode = Euler2D.TANGENT,
    gas = DRY_AIR,
    scale = scale,
    show_detailed_info = false,
    info_frequency = 250,
    write_frequency = 500,
    max_tsteps = 30000,
    output_tag = "minimum_working_example/tangent_mode_5k",
    output_channel_size = 2,
    tasks_per_axis = 12,
    boundary_conditions = bcs,
)

Euler2D.start_simulation_from_initial_conditions(
    u0,
    starting_parameters,
    25.0,
    probe,
    bounds,
    ncells,
    bcs,
    sim_config,
);
