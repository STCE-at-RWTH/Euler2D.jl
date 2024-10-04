using Euler2D
using LinearAlgebra
using ShockwaveProperties
using StaticArrays
using Unitful

function launder_units(pp)
    c1 = ConservedProps(pp, DRY_AIR)
    v1 = state_to_vector(c1)
    return ConservedProps(v1)
end

ρL = 1.0u"kg/m^3"
vL = [0.0u"m/s", 0.0u"m/s"]
PL = 10.0u"Pa"
TL = uconvert(u"K", PL / (ρL * DRY_AIR.R))
ML = vL / speed_of_sound(ρL, PL, DRY_AIR)

ρR = 0.125 * ρL
vR = [0.0u"m/s", 0.0u"m/s"]
PR = 0.1 * PL
TR = uconvert(u"K", PR / (ρR * DRY_AIR.R))
MR = vR / speed_of_sound(ρR, PR, DRY_AIR)

s_high = ConservedProps(PrimitiveProps(ρL, [ML[1]], TL), DRY_AIR)
s_low = ConservedProps(PrimitiveProps(ρR, [MR[2]], TR), DRY_AIR)

u0_1d(x) = state_to_vector(x < 0.5 ? s_high : s_low)

extrapolation_bcs = EdgeBoundary(ExtrapolateToPhantom(), ExtrapolateToPhantom())
bcs_1d = (extrapolation_bcs,)

bounds_x = (0.0, 2.0)

ncells_x = 500

##

# simulation 1
simulate_euler_equations(
    u0_1d,
    0.1,
    bcs_1d,
    (bounds_x,),
    (ncells_x,);
    cfl_limit = 0.75,
    output_tag = "sod_shock_right_1d",
)

# simulation 2
simulate_euler_equations(
    u0_1d,
    0.1,
    bcs_1d,
    (bounds_x,),
    (ncells_x,);
    cfl_limit = 0.75,
    output_tag = "sod_shock_right_1d",
)
