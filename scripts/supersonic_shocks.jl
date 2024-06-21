using Euler2D
using LinearAlgebra
using ShockwaveProperties
using Unitful

##
# SHOCK SCENARIO ONE
# SHOCK AT X = 0
# SUPERSONIC FLOW IMPACTS STATIC ATMOSPHERIC AIR

uL_1 = ConservedProps(PrimitiveProps(1.225, [1.5], 300.0); gas = DRY_AIR)
uR_1 = ConservedProps(PrimitiveProps(1.225, [0.0], 350.0); gas = DRY_AIR)

bounds_1 = (-25.0, 225.0)
bcs_1 = EdgeBoundary(SupersonicInflow(uL_1, DRY_AIR), ExtrapolateToPhantom())

simulate_euler_equations(
    0.2,
    (bcs_1,),
    (bounds_1,),
    (500,);
    output_tag = "supersonic_shock_1",
) do x
    res = x < 0 ? uL_1 : uL_1
    state_to_vector(res)
end

##
# SHOCK SCENARIO TWO
# SHOCKS AT X = -50 and X = 50
# SUPERSONIC INFLOW ON BOTH SIDES

uL_2 = ConservedProps(PrimitiveProps(1.225, [1.5], 300.0); gas = DRY_AIR)
uM_2 = ConservedProps(PrimitiveProps(1.225, [0.0], 350.0); gas = DRY_AIR)
uR_2 = ConservedProps(PrimitiveProps(1.225, [-1.5], 300.0); gas = DRY_AIR)

bcs_2 = EdgeBoundary(SupersonicInflow(uL_2, DRY_AIR), SupersonicInflow(uR_2, DRY_AIR))
simulate_euler_equations(
    0.2,
    (bcs_1,),
    ((-100.0, 100.0),),
    (500,);
    output_tag = "supersonic_shock_2",
) do x
    res = if x < -50
        uL_2
    elseif x > 50
        uR_2
    else
        uM_2
    end
    state_to_vector(res)
end

# SCENARIO THREE
# MASS ACCUMULATES

bcs_3 = EdgeBoundary(SupersonicInflow(uL_1, DRY_AIR), StrongWall())
simulate_euler_equations(
    0.2,
    (bcs_3,),
    ((0.0, 200.0),),
    (1000,);
    output_tag = "supersonic_shock_3",
) do x
    state_to_vector(uL_1)
end