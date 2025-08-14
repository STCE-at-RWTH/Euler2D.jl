# Method skeleton for the exact Riemann solver.
# if NonlinearSolve.jl is available see <package root>/ext/ExactRiemannExt
# for actual implementation details
#
# the stubs are here to remind you to install NonlinearSolve.jl, which is a hefty dependency for a feature you may not need

# following the derviations in Smoller 1994
# Shock Waves and Reaction-Diffisuion Equations

"""Pressure ratio abbreviated pi"""
function _π(uL, uR, gas)
    return dimensionless_pressure(uR, gas) / dimensionless_pressure(uL, gas)
end

"""Density ratio abbreviated z"""
_z(uL, uR) = uR[1] / uL[1]

_β(gas) = (gas.γ + 1) / (gas.γ - 1)
_τ(gas) = (gas.γ - 1) / (2 * gas.γ)
_x(uL, uR, gas) = -log(_π(uL, uR, gas))

density_bound_uR(uL, gas) = (uL[1], _β(gas) * uL[1])

function jump_ratios(uL, uR, gas)
    A = _z(uL, uR)
    B = _π(uL, uR, gas)
    cL = dimensionless_speed_of_sound(uL, gas)
    C = (uR[2] / uR[1] - uL[2] / uL[1]) / cL
    return (A, B, C)
end

function one_shock_ratios(x, gas)
    pressure_ratio = exp(-x)
    density_ratio = (exp(x) + _β(gas)) / (1 + _β(gas) * exp(x))
    velocity_jump_speed_of_sound_ratio =
        ((2 * sqrt(_τ(gas))) / (gas.γ - 1) * (1 - exp(-x)) / (sqrt(1 + _β(gas) * exp(-x))))
    return (pressure_ratio, density_ratio, velocity_jump_speed_of_sound_ratio)
end

function one_rarefaction_ratios(x, gas)
    pressure_ratio = exp(-x)
    density_ratio = exp(-x / gas.γ)
    velocity_jump_speed_of_sound_ratio = 2 / (gas.γ - 1) * (1 - exp(-_τ(gas) * x))
    return (pressure_ratio, density_ratio, velocity_jump_speed_of_sound_ratio)
end

function one_family(x, gas)
    return x < 0.0 ? one_shock_ratios(x, gas) : one_rarefaction_ratios(x, gas)
end

function two_family(x, gas)
    return (1.0, exp(x), 0.0)
end

function test_for_vacuum_in_solution(uL, uR, gas)
    (cL, cR) = dimensionless_speed_of_sound.((uL, uR), gas)
    vL = uL[2] / uL[1]
    vR = uR[2] / uR[1]
    return vR - vL ≥ 2 / (gas.γ - 1) * (cL + cR)
end

function test_1_family_for_rarefaction(uL, uR, gas)
    (A, B, C) = jump_ratios(uL, uR, gas)
end
