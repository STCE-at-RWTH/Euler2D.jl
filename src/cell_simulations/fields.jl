# methods for computing the various scalar fields of a cell based simulation

"""
    density_field(csim::CellBasedEulerSim, n)

Compute the density field for a cell-based Euler simulation `csim` at time step `n`.
"""
function density_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    ρ = Array{Union{T, Nothing}, 2}(undef, grid_size(csim))
    fill!(ρ, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρ[i] = u[1]
    end
    return ρ
end

"""
    density_field(csim::CellBasedEulerSim, scale, n)

Compute the density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function density_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        scale::EulerEqnsScaling,
    ) where {T}
    return map(density_field(csim, n)) do ρ
        isnothing(ρ) && return nothing
        return ρ * density_scale(scale)
    end
end

"""
    momentum_density_field(csim::CellBasedEulerSim, n)

Compute the dimensionless momentum density field for a cell-based Euler simulation `csim` at time step `n`.
"""
function momentum_density_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    ρv = Array{T, 3}(undef, (2, grid_size(csim)...))
    fill!(ρv, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρv[:, i] = select_middle(u)
    end
    return ρv
end

"""
    momentum_density_field(csim::CellBasedEulerSim, scale, n)

Compute the momentum density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function momentum_density_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        scale::EulerEqnsScaling,
    ) where {T}
    return map(momentum_density_field(csim, n)) do ρv
        isnothing(ρv) && return nothing
        return ρv * density_scale(scale) * velocity_scale(scale)
    end
end

"""
    velocity_field(csim::CellBasedEulerSim, n)

Compute the dimensionless velocity field for a cell-based Euler simulation `csim` at time step `n`.
"""
function velocity_field(csim::CellBasedEulerSim{T}, n::Integer) where {T}
    _, u_cells = nth_step(csim, n)
    v = Array{Union{T, Nothing}, 3}(undef, (2, grid_size(csim)...))
    fill!(v, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        v[:, i] = select_middle(u) / first(u)
    end
    return v
end

"""
    velocity_field(csim::CellBasedEulerSim, scale, n)

Compute the velocity field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function velocity_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        scale::EulerEqnsScaling,
    ) where {T}
    return map(velocity_field(csim, n)) do v
        isnothing(v) && return nothing
        return v * velocity_scale(scale)
    end
end

"""
    total_internal_energy_density_field(csim::CellBasedEulerSim, n)

Compute the dimensionless total internal energy density field for a cell-based Euler simulation at time step `n`.
"""
function total_internal_energy_density_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
    ) where {T}
    _, u_cells = nth_step(csim, n)
    ρE = Array{Union{T, Nothing}, 2}(undef, grid_size(csim))
    fill!(ρE, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        ρE[i] = last(u)
    end
    return ρE
end

"""
    total_internal_energy_density_field(csim::CellBasedEulerSim, scale, n)

Compute the total internal energy density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function total_internal_energy_density_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        scale::EulerEqnsScaling,
    ) where {T}
    return map(total_internal_energy_density_field(csim, n)) do ρE
        isnothing(ρE) && return nothing
        return ρE * energy_density_scale(scale)
    end
end

"""
    pressure_field(csim::CellBasedEulerSim, n, gas)

Compute the dimensionless pressure field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function pressure_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        gas::CaloricallyPerfectGas,
    ) where {T}
    _, u_cells = nth_step(csim, n)
    P = Array{Union{T, Nothing}, 2}(undef, grid_size(csim))
    fill!(P, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        P[i] = dimensionless_pressure(u, gas)
    end
    return P
end

"""
    pressure_field(csim::CellBasedEulerSim, scale, n)

Compute the total internal energy density field for a cell-based Euler simulation `csim` and redimensionalize it at time step `n`.
"""
function pressure_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        gas::CaloricallyPerfectGas,
        scale::EulerEqnsScaling,
    ) where {T}
    return map!(pressure_field(csim, n, gas)) do P
        isnothing(P) && return nothing
        return P * pressure_scale(scale)
    end
end

"""
    mach_number_field(csim::CellBasedEulerSim, n, gas)

Compute the Mach number field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.
"""
function mach_number_field(
        csim::CellBasedEulerSim{T},
        n::Integer,
        gas::CaloricallyPerfectGas,
    ) where {T}
    _, u_cells = nth_step(csim, n)
    # is this a huge runtime problem? who can know.
    M = Array{Union{T, Nothing}, 3}(undef, (2, grid_size(csim)...))
    fill!(M, nothing)
    for i in eachindex(IndexCartesian(), csim.cell_ids)
        csim.cell_ids[i] == 0 && continue
        u = u_cells[csim.cell_ids[i]].u
        a = dimensionless_speed_of_sound(u, gas)
        M[:, i] = select_middle(u) / (first(u) * a)
    end
    return M
end

# for completeness
# the scaling doesn't actually matter for the mach number
mach_number_field(
    csim::CellBasedEulerSim,
    n::Integer,
    gas::CaloricallyPerfectGas,
    ::EulerEqnsScaling,
) = mach_number_field(csim, n, gas)

function norm_mach_number_field(csim::CellBasedEulerSim{T}, n::Integer, gas, args...) where {T}
    M = mach_number_field(csim, n, gas, args...)
    return map(eachslice(M; dims = (2, 3))) do m
        any(isnothing, m) && return nothing
        return norm(m)
    end
end
