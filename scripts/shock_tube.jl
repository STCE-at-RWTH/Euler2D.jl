using Euler2D
using LinearAlgebra
using ShockwaveProperties
using Unitful

# we want to set up Sod's shock tube problem

ρL = 1.0u"kg/m^3"
vL = [0.0u"m/s", 0.0u"m/s"]
PL = 10.0u"Pa"
TL = uconvert(u"K", PL / (ρL * DRY_AIR.R))
ML = vL / speed_of_sound(ρL, PL; gas = DRY_AIR)

ρR = 0.125 * ρL
vR = [0.0u"m/s", 0.0u"m/s"]
PR = 0.1 * PL
TR = uconvert(u"K", PR / (ρR * DRY_AIR.R))
MR = vR / speed_of_sound(ρR, PR; gas = DRY_AIR)

s_high = ConservedProps(PrimitiveProps(ρL, [ML[1]], TL); gas = DRY_AIR)
s_low = ConservedProps(PrimitiveProps(ρR, [MR[2]], TR); gas = DRY_AIR)

s_high_2d = ConservedProps(PrimitiveProps(ρL, ML, TL); gas = DRY_AIR)
s_low_2d = ConservedProps(PrimitiveProps(ρR, MR, TR); gas = DRY_AIR)

u0_1d(x) = state_to_vector(x < 0.5 ? s_high : s_low)
u0_2d(x, y) = state_to_vector(x < 0.5 ? s_high_2d : s_low_2d)

extrapolation_bcs = EdgeBoundary(ExtrapolateToPhantom(), ExtrapolateToPhantom())
bcs_1d = (extrapolation_bcs,)
bcs_2d = (extrapolation_bcs, extrapolation_bcs)

bounds_x = (0.0, 1.0)
bounds_y = (0.0, 1.0)

ncells_x = 500
ncells_y = 10

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
u1_1d(x) = state_to_vector(x < 0.5 ? s_low : s_high)

simulate_euler_equations(
    u1_1d,
    0.1,
    bcs_1d,
    (bounds_x,),
    (ncells_x,);
    cfl_limit = 0.75,
    output_tag = "sod_shock_left_1d",
)

# simulation 3 (quasi 1d)

simulate_euler_equations(
    u0_2d,
    0.1,
    bcs_2d,
    (bounds_x, bounds_y),
    (ncells_x, ncells_y);
    cfl_limit = 0.75,
    output_tag = "sod_shock_right_2d",
)