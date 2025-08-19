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
    @test u1_star[2] / u1_star[1] ≈ 2.0
    @test a1_star ≈ 1.0
end

@testset "Exact Riemann Solver" begin
    using Euler2D: ExactRiemannSolverInternal

    # test for states only in x-axis
    ux_1 = @SVector [1.0, 1.0, 0.0, 6000.0]
    ux_2 = @SVector [1.0, 0.5, 0.0, 6000.0]
    ux_3 = @SVector [2.0, 1.0, 0.0, 6000.0]
    ux_4 = @SVector [1.0, -1.0, 0.0, 6000.0]
    ux_5 = @SVector [2.0, -1.0, 0.0, 6000.0]

    @test all(
        ExactRiemannSolverInternal.jump_ratios(ux_1, ux_1, DRY_AIR) .== (1.0, 1.0, 0.0),
    )
    @test all(
        x -> isapprox(x, 0.0; atol = 1.0e-12),
        ExactRiemannSolverInternal.solve_for_jump_parameters(ux_1, ux_1, DRY_AIR),
    )
    @testset for (uL, uR) ∈
                 [(ux_1, ux_2), (ux_2, ux_1), (ux_1, ux_4), (ux_4, ux_5), (ux_4, ux_2)]
        (x1, x2, x3) = ExactRiemannSolverInternal.solve_for_jump_parameters(uL, uR, DRY_AIR)
        primL = ExactRiemannSolverInternal.to_ρPv(uL, DRY_AIR)
        primL_★ = ExactRiemannSolverInternal.T_1(primL..., x1, DRY_AIR)
        primR_★ = ExactRiemannSolverInternal.T_2(primL_★..., x2, DRY_AIR)
        primR_appx = ExactRiemannSolverInternal.T_3(primR_★..., x3, DRY_AIR)
        primR = ExactRiemannSolverInternal.to_ρPv(uR, DRY_AIR)
        @test primR_appx[1] ≈ primR[1]
        @test primR_appx[2] ≈ primR[2]
        @test primR_appx[3] ≈ primR[3]

        three_wave = ExactRiemannSolverInternal.test_3_wave_for_rarefaction(uL, uR, DRY_AIR)
        @test isapprox(x3, 0.0; atol = 1.0e-12) ||
              ((three_wave && x3 < 0) && (!three_wave && x3 > 0))
    end
end
