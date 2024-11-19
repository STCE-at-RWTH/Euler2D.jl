using Euler2D
using ShockwaveProperties
using StaticArrays
using Test
using Unitful

@testset "Nondimensionalisation" begin
    using Euler2D: EulerEqnsScaling, _SI_DEFAULT_SCALE
    using Euler2D: nondimensionalize, redimensionalize
    using Euler2D: dimensionless_speed_of_sound, dimensionless_total_enthalpy_density

    u1 = ConservedProps(PrimitiveProps(0.1u"kg/m^3", SVector(2.0, 0.0), 225.0u"K"), DRY_AIR)
    a = speed_of_sound(u1, DRY_AIR)
    scale = EulerEqnsScaling(1.0u"m", 1.0u"kg/m^3", a)

    u1_star = nondimensionalize(u1, scale)
    a1_star = dimensionless_speed_of_sound(u1_star, DRY_AIR)
    @test u1_star[2]/u1_star[1] ≈ 2.0
    @test a1_star ≈ 1.0
end