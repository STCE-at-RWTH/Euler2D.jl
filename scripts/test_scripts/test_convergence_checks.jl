using BenchmarkTools

using Euler2D
using ShockwaveProperties
using Unitful
using LinearAlgebra
using StaticArrays

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

const bcs = (
    ExtrapolateToPhantom(), # north 
    StrongWall(), # south
    ExtrapolateToPhantom(), # east
    ExtrapolateToPhantom(), # west
    StrongWall(), # walls
)

bounds = ((-1.5, 0.0), (0.0, 2.0))
probe = [
    CircularObstacle((0.0, 0.0), 0.75),
    RectangularObstacle(SVector(1.0, 0.0), SVector(2.0, 1.5)),
]
ncells = (540, 540 * 4 ÷ 3)
