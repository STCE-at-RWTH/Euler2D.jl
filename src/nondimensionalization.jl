using ShockwaveProperties
using Unitful

struct CharacteristicScales{T, Q1 <: Length{T}, Q2 <: Velocity{T}, Q3 <: Pressure{T}}
    ρ_0::Q1
    v_0::Q2
    P_0::Q3
end

function nondimensionalize(u::ConservedProps{T}, scale, gas::CaloricallyPerfectGas)
    ρ_star = u.ρ / scale.ρ_0
    ρv_star = u.ρv / (scale.ρ_0 * scale.v_0)
    ρe = static_internal_energy_density(u, gas)
    ρe_star = uconvert(Unitful.NoUnits, ρe / scale.P_0)
    ρke_star = ρv_star ⋅ ρv_star / ρ_star
    ρE_star = ρe_star + 
    return SVector{4, T}(u.ρ/scale.ρ_0, (u.ρv/(scale.ρ_0*scale.v_0))..., )
end
