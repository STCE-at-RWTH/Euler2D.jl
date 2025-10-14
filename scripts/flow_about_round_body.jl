# using Profile
# using PProf

using Euler2D
using LinearAlgebra
using Unitful
using ShockwaveProperties
using StaticArrays

"""
    u0(x, p)

Accepts ``x∈ℝ^2`` and a vector of three parameters: free stream density, mach number, and temperature (understood in metric base units)
"""
function u0(x, p)
    pp = PrimitiveProps(p[1], SVector(p[2], 0.0), p[3])
    return ConservedProps(pp, DRY_AIR)
end

starting_parameters = SVector(0.662, 4.0, 220.0)
ambient = u0(nothing, starting_parameters)

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
const ncells = (540, 540 * 4 ÷ 3)

# Profile.init(; n = 10^7, delay = 0.005)
# Profile.clear()

Euler2D.simulate_euler_equations_cells(
    u0,
    starting_parameters,
    20.0,
    bcs,
    probe,
    bounds,
    ncells;
    mode = Euler2D.TANGENT,
    gas = DRY_AIR,
    scale = scale,
    show_detailed_info = false,
    info_frequency = 1,
    write_frequency = 25,
    max_tsteps = 200,
    output_tag = "validate_new_partitioning/validate_update_fusion",
    output_channel_size = 2,
    tasks_per_axis = 16,
);
