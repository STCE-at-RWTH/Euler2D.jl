# IMPLEMENTATION OF
# Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows  
# Takeshi R. Fujimoto ∗, Taro Kawasaki 1, Keiichi Kitamura
# Journal of Computational Physics, 396, pp. 264 - 279
#

module CannyShockSensor

using LinearAlgebra
using ShockwaveProperties
using StaticArrays
using Tullio

using Euler2D
using Euler2D: CellBasedEulerSim, select_middle
using PlanePolygons

export find_shock_in_timestep

_diff_op(T) = SVector{3,T}(-one(T), zero(T), one(T))
_avg_op(T) = SVector{3,T}(one(T), 2 * one(T), one(T))

_sobel_X(T) = _diff_op(T) * _avg_op(T)'
_sobel_Y(T) = _avg_op(T) * _diff_op(T)'

# don't ask.
# makes the whole "symmetry about y=0" thing work
# WILL NOT WORK WITH OFFSETARRAYS
function _pad_by_copying_outwards!(target, pad_size)
    for i = pad_size-1:-1:0
        @views begin
            target[begin+i, :] = target[begin+i+1, :]
            target[end-i, :] = target[end-i-1, :]
            target[:, begin+i] = target[:, begin+i+1]
            target[:, end-i] = target[:, end-i-1]
        end
    end
    return nothing
end

_convolve_op(mat, op) = @tullio out[i, j] := mat[i+k+1, j+l+1] * op[k+2, l+2]
_convolve_op!(res, mat, op) = @tullio res[i, j] = mat[i+k+1, j+l+1] * op[k+2, l+2]

_d_dx_convolve(mat) = _convolve_op(mat, _sobel_X(eltype(mat)))
_d_dy_convolve(mat) = _convolve_op(mat, _sobel_Y(eltype(mat)))
_d_dx_convolve!(res, mat) = _convolve_op!(res, mat, _sobel_X(eltype(mat)))
_d_dy_convolve!(res, mat) = _convolve_op!(res, mat, _sobel_Y(eltype(mat)))

"""
    convolve_sobel(field::Matrix{T})

Computes the convolution of the Sobel gradient kernel ``G_x`` and ``G_y`` with the given field.
"""
function convolve_sobel(field)
    sX = _sobel_X(eltype(field))
    sY = _sobel_Y(eltype(field))
    outX = _convolve_op(field, sX)
    outY = _convolve_op(field, sY)
    return outX, outY
end

gradient_direction(Gx, Gy) = atan(Gy / Gx)

function discretize_gradient_direction(θ)
    if -π / 8 ≤ θ < π / 8
        return 0
    elseif π / 8 ≤ θ < 3 * π / 8
        return π / 4
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return π / 2
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return 3 * π / 4
    elseif 7 * π / 8 ≤ θ
        return π
    elseif -3 * π / 8 ≤ θ < -π / 8
        return -π / 4
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return -π / 2
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return -3π / 4
    elseif θ < -7 * π / 8
        return -π
    end
end

function gradient_grid_direction(θ)
    if -π / 8 ≤ θ < π / 8
        return CartesianIndex(1, 0)
    elseif π / 8 ≤ θ < 3 * π / 8
        return CartesianIndex(1, 1)
    elseif 3 * π / 8 ≤ θ < 5 * π / 8
        return CartesianIndex(0, 1)
    elseif 5 * π / 8 ≤ θ < 7 * π / 8
        return CartesianIndex(-1, 1)
    elseif 7 * π / 8 ≤ θ
        return CartesianIndex(-1, 0)
    elseif -3 * π / 8 ≤ θ < -π / 8
        return CartesianIndex(1, -1)
    elseif -5 * π / 8 ≤ θ < -3 * π / 8
        return CartesianIndex(0, -1)
    elseif -7 * π / 8 ≤ θ < -5 * π / 8
        return CartesianIndex(-1, -1)
    elseif θ < -7 * π / 8
        return CartesianIndex(-1, 0)
    end
end

"""
    is_edge_candidate(dP2_view, Gx, Gy)

Checks if the center value of `dP2_view` is a maximum in direction `θ`.
"""
function is_edge_candidate(dP2_view, θ)
    @assert size(dP2_view) == (3, 3)
    grid_theta = gradient_grid_direction(θ)
    local idx = CartesianIndex(2, 2)
    return dP2_view[idx+grid_theta] < dP2_view[idx] &&
           dP2_view[idx-grid_theta] < dP2_view[idx]
end

function _shock_velocity(θij, u1, u2, gas)
    h1 = dimensionless_enthalpy(u1, gas)
    h2 = dimensionless_enthalpy(u2, gas)
    v1 = dimensionless_velocity(u1)
    v2 = dimensionless_velocity(u2)
    Δv = v1 - v2
    A = 2 * h1 + v1 ⋅ v1
    B = 2 * h2 + v2 ⋅ v2
    vs_x = (A - B) / (2 * (Δv[1] + tan(θij) * Δv[2]))
    vs_y = (A - B) / (2 * (cot(θij) * Δv[1] + Δv[2]))
    return SVector(vs_x, vs_y)
end

"""
    hugoniot_equation_relative_err(u_A, u_B, gas)

Compute the relative error between the two sides of the Hugoniot equation, relative to the change in internal energy ``e``.
"""
function hugoniot_equation_relative_err(u_A, u_B, gas)
    # try to apply hugoniot equation
    # to test for admissible discontinuity?
    e_A = dimensionless_internal_energy(u_A)
    e_B = dimensionless_internal_energy(u_B)
    P_A = dimensionless_pressure(u_A, gas)
    P_B = dimensionless_pressure(u_B, gas)
    ρ_A = u_A[begin]
    ρ_B = u_B[begin]
    # if we get the order wrong, we just mutiply this by -1... should be fine?
    lhs = e_B - e_A
    rhs = (P_A + P_B) / 2 * (inv(ρ_A) - inv(ρ_B))
    return abs((lhs - rhs) / lhs)
end

"""
    mach_number_change_across_shock(θij, u1, u2, gas)

Compute the Mach number jump error and a smoothness criterion between two states `u1` and `u2`.

1. Translate the states `u1` and `u2` into a coordinate system that moves with the shock.
`u1` is upstream, `u2` is downstream. In the bow-shock problem, we want to assert that there is
a pressure increase across the shock (upstream -> downstream).
2. Estimate the change across that shock. Note that `a` does not change under a velocity transformation
3. Return the relative change in the mach number, which is an estimate of the "strength" of the shock.
"""
function mach_number_change_across_shock(θij, u_A, u_B, gas)
    v_shock = _shock_velocity(θij, u_A, u_B, gas)
    u_shock_A = shift_velocity_coordinates(u_A, v_shock)
    u_shock_B = shift_velocity_coordinates(u_B, v_shock)
    n = normalize(v_shock)
    Mn_A = dimensionless_mach_number(u_shock_A, gas) ⋅ n
    Mn_B = dimensionless_mach_number(u_shock_B, gas) ⋅ n
    return abs((Mn_A - Mn_B) / Mn_A)
end

"""
    ShockSensorInfo

Fields
---
- `candidates`: Which cell ids are candidate cells for a shock?
- `n_candidates`: How many are there?
- `n_thinned`: How many cells were candidates after edge thinning?
- `n_rejected_smooth`: How many of the thinned edges failed to satisfy the non-smoothness threshold?
- `n_rejected_rh`: How many of the thinned edges failed to satisfy the Hugoniot equation?
"""
struct ShockSensorInfo
    candidates::Matrix{Bool}
    n_candidate_cells::Int
    n_thinned::Int
    rh_relative_error_max::Float64
    smoothness_thold::Float64
    n_rejected_smooth::Int
    n_rejected_rh::Int
end

function find_shock_in_timestep(
    sim::CellBasedEulerSim{T,C},
    t,
    gas;
    rh_rel_error_max = 0.5,
    continuous_variation_thold = 0.01,
) where {T,C}
    # TODO really gotta figure out how to deal with nothings or missings in this matrix
    # TODO this solution is not very good
    pfield = Matrix{T}(undef, grid_size(sim) .+ 4)
    pfield_nopad = @view pfield[begin+2:end-2, begin+2:end-2]
    @assert size(pfield_nopad) == size(sim.cell_ids)
    Euler2D.pressure_field!(pfield_nopad, sim, t, gas)
    _pad_by_copying_outwards!(pfield, 2)
    # find the gradients via the sobel operators
    # not quite accurate but good enough since we discretize to the compass directions anyway
    dPdx = _d_dx_convolve(pfield)
    dPdy = _d_dy_convolve(pfield)
    dP2 = dPdx .^ 2 .+ dPdy .^ 2
    @assert size(dPdx) == size(dPdy) == (size(sim.cell_ids) .+ 2)
    dPdx_nopad = @view dPdx[begin+1:end-1, begin+1:end-1]
    dPdy_nopad = @view dPdy[begin+1:end-1, begin+1:end-1]
    candidates = Matrix{Bool}(undef, grid_size(sim))
    for i ∈ eachindex(IndexCartesian(), candidates, dPdx_nopad, dPdy_nopad)
        window = i:(i+CartesianIndex(2, 2))
        θ = atan(dPdy_nopad[i], dPdx_nopad[i])
        candidates[i] = is_edge_candidate(@view(dP2[window]), θ)
    end
    n_marked_edges = count(candidates)
    n_too_smooth = 0
    n_no_shock = 0

    for i ∈ eachindex(
        IndexCartesian(),
        sim.cell_ids,
        candidates,
        pfield_nopad,
        dPdx_nopad,
        dPdy_nopad,
    )
        if !candidates[i]
            continue
        end
        θ = atan(dPdy_nopad[i], dPdx_nopad[i])
        θij = discretize_gradient_direction(θ)
        Δi = gradient_grid_direction(θ)
        # make sure we can look up i±θij and that neither of those cells are empty
        if !(
            (checkbounds(Bool, candidates, i + Δi) && sim.cell_ids[i+Δi] != 0) &&
            (checkbounds(Bool, candidates, i - Δi) && sim.cell_ids[i-Δi] != 0)
        )
            candidates[i] = false
            n_no_shock += 1
            continue
        end

        _, cells = nth_step(sim, t)
        u_A = cells[sim.cell_ids[i-Δi]].u
        u_B = cells[sim.cell_ids[i+Δi]].u
        abs_relative_err = hugoniot_equation_relative_err(u_A, u_B, gas)
        if abs_relative_err >= rh_rel_error_max
            candidates[i] = false
            n_no_shock += 1
            continue
        end
        abs_smoothness_err = mach_number_change_across_shock(θij, u_A, u_B, gas)
        if abs_smoothness_err <= continuous_variation_thold
            candidates[i] = false
            n_too_smooth += 1
        end
    end

    return ShockSensorInfo(
        candidates,
        count(candidates),
        n_marked_edges,
        rh_rel_error_max,
        continuous_variation_thold,
        n_too_smooth,
        n_no_shock,
    )
end

"""
    extract_bow_shock_points(sim, sensor_info)

Tries to extract the main bow shock from a simulation given the results of applying a shock sensor.
Will throw out any cells that deviate from both of their immediate neighbors by more than `gap_size` cell radii.
"""
function extract_bow_shock_points(sim, sensor_info; gap_size = 8)
    _, cells = nth_step(sim, 1)
    size_diff = grid_size(sim) .- size(sensor_info.candidates)
    @assert all(iseven.(size_diff))
    Δi, Δj = div.(size_diff, 2)
    ids_view = @view sim.cell_ids[begin+Δi:end-Δi, begin+Δj:end-Δj]
    @assert size(ids_view) == size(sensor_info.candidates)
    @assert size(ids_view) .+ size_diff == grid_size(sim)
    max_gap_length = gap_size * norm(Euler2D.minimum_cell_size(sim))
    candidate_points = map(
        zip(
            eachslice(sensor_info.candidates; dims = (2,)),
            eachslice(ids_view; dims = (2,)),
        ),
    ) do (candidate_row, ids_row)
        idx = findfirst(candidate_row)
        isnothing(idx) && return SVector(Inf, Inf)
        cell_id = ids_row[idx]
        return cells[cell_id].center
    end
    filter!(≠(SVector(Inf, Inf)), candidate_points)
    return candidate_points
end

end
