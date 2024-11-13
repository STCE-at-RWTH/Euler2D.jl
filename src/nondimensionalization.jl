struct CharacteristicScales{T, Q1 <: Length{T}, Q2 <: Velocity{T}, Q3 <: Pressure{T}}
    ρ_0::Q1
    v_0::Q2
    P_0::Q3
end

