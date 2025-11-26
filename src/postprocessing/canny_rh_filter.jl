# IMPLEMENTATION OF
# Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows  
# Takeshi R. Fujimoto ∗, Taro Kawasaki 1, Keiichi Kitamura
# Journal of Computational Physics, 396, pp. 264 - 279
#
# Altered to compare the Hugoniot equation values rather than the mach number ratio.

module CannyShockSensor

using LinearAlgebra

using Accessors
using Interpolations
using OhMyThreads
using StaticArrays
using Tullio

using Euler2D
using Euler2D: translate_velocity_coordinates, change_space_coordinates_and_shift_v
using PlanePolygons
using ShockwaveProperties

export find_shock_in_timestep, extract_bow_shock_interpolation

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
    relative_mach_number_change_as_if_shock(θij, u1, u2, gas)

Compute the Mach number jump error and a smoothness criterion between two states `u1` and `u2`.

1. Translate the states `u1` and `u2` into a coordinate system that moves with the shock.
`u1` is upstream, `u2` is downstream. In the bow-shock problem, we want to assert that there is
a pressure increase across the shock (upstream -> downstream).
2. Estimate the change across that shock. Note that `a` does not change under a velocity transformation
3. Return the relative change in the mach number, which is an estimate of the "strength" of the shock.
"""
function relative_mach_number_change_as_if_shock(θij, u_A, u_B, gas)
    v_shock = _shock_velocity(θij, u_A, u_B, gas)
    u_shock_A = translate_velocity_coordinates(u_A, v_shock)
    u_shock_B = translate_velocity_coordinates(u_B, v_shock)
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

"""
    _pressure_field!(result, csim::CellBasedEulerSim, n, gas)

Compute the dimensionless pressure field for a cell-based Euler simulation `csim` at time step `n` in gas `gas`.

Stores zero if the cell in question is not active.
Stores results in `result`. Returns `result`.
"""
function _pressure_field!(
    result,
    csim::CellBasedEulerSim,
    n::Integer,
    gas::CaloricallyPerfectGas,
)
    _, cells = nth_step(csim, n)
    for i ∈ eachindex(result, csim.cell_ids)
        if csim.cell_ids[i] == 0
            result[i] = zero(eltype(result))
        else
            result[i] = dimensionless_pressure(cells[csim.cell_ids[i]].u, gas)
        end
    end
    # TODO really gotta figure out how to deal with nothings or missings in this matrix
    # TODO this solution is not very good
    return result
end

function _fill_pfield_and_pad!(pfield, pad_size, sim, t, gas)
    p_nopad = @view pfield[(begin+pad_size):(end-pad_size), (begin+pad_size):(end-pad_size)]
    @assert size(p_nopad) == size(sim.cell_ids)
    _pressure_field!(p_nopad, sim, t, gas)
    _pad_by_copying_outwards!(pfield, pad_size)

    return pfield
end

function find_shock_in_timestep(
    sim::CellBasedEulerSim{T,C},
    t,
    gas;
    rh_rel_error_max = 0.5,
    continuous_variation_thold = 0.05,
) where {T,C}
    N_pad = 2
    # padding size is 2 on either size, hence we need 2*N_pad extra values
    pfield = Matrix{T}(undef, grid_size(sim) .+ 2 * N_pad)
    pfield_nopad = @view pfield[(begin+N_pad):(end-N_pad), (begin+N_pad):(end-N_pad)]
    _fill_pfield_and_pad!(pfield, N_pad, sim, t, gas)

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

    @assert (
        size(sim.cell_ids) ==
        size(candidates) ==
        size(pfield_nopad) ==
        size(dPdx_nopad) ==
        size(dPdy_nopad)
    )

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
        abs_smoothness_err = relative_mach_number_change_as_if_shock(θij, u_A, u_B, gas)
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

Extracts the center of each of the first marked shock cells in the "row".
"""
function extract_bow_shock_points(sim, sensor_info; gap_size = 8)
    _, cells = nth_step(sim, 1)
    size_diff = grid_size(sim) .- size(sensor_info.candidates)
    Δi, Δj = div.(size_diff, 2)
    ids_view = @view sim.cell_ids[begin+Δi:end-Δi, begin+Δj:end-Δj]
    @assert size(ids_view) == size(sensor_info.candidates)
    @assert size(ids_view) .+ size_diff == grid_size(sim)
    # TODO maybe implement gap finding?
    # max_gap_length = gap_size * norm(Euler2D.minimum_cell_size(sim))
    candidate_points = map(
        eachslice(sensor_info.candidates; dims = (2,)),
        eachslice(ids_view; dims = (2,)),
    ) do candidate_row, ids_row
        idx = findfirst(candidate_row)
        isnothing(idx) && return SVector(Inf, Inf)
        cell_id = ids_row[idx]
        return cells[cell_id].center #- SVector(0.0, cells[cell_id].extent[2] / 2)
    end
    filter!(≠(SVector(Inf, Inf)), candidate_points)
    return candidate_points
end

function extract_bow_shock_interpolation(sim, sensor_info; skip_pts = 10)
    points = extract_bow_shock_points(sim, sensor_info)[begin:skip_pts:end]
    return scale(
        interpolate(points, BSpline(Linear())),
        range(0.0, 1.0; length = length(points)),
    )
end

function _pts_via_range_interp(xs, yrange, y_actual)
    interp = scale(interpolate(xs, BSpline(Linear())), yrange)
    x_actual = interp.(y_actual)
    return x_actual
end

function _pts_via_gridded_interp(xs, ys, y_actual)
    interp = interpolate((ys,), xs, Gridded(Linear()))
    x_actual = interp.(y_actual)
    return x_actual
end

"""
    NPointBowShockInterpolation

Represents specifically an interpolation of the points generated by the y-axis symmetric bow shock problem.

Hüser was heckin' right.
"""
struct NPointBowShockInterpolation{YT<:AbstractVector,XT<:AbstractVector}
    N::Int
    ys::YT
    xs::XT
end

function NPointBowShockInterpolation(N, sim, sensor_info)
    shock_points = extract_bow_shock_points(sim, sensor_info)
    # speed penalty on a reinterpret here should be ok
    shock_points_r = reinterpret(reshape, Float64, shock_points)
    shock_ys = @view shock_points_r[2, :]
    shock_xs = @view shock_points_r[1, :]

    ymin = 0.0 # just in case we want to change this later?
    ymax = maximum(shock_ys)
    half_dy = (ymax - ymin) / (2 * N - 1)
    actual_ys = range(half_dy, ymax; length = N)
    actual_xs = if length(shock_ys) == length(cell_centers(sim, 2))
        # can use fast interpolation!
        _pts_via_range_interp(shock_xs, cell_centers(sim, 2), actual_ys)
    else
        #slow interpolation
        _pts_via_gridded_interp(shock_xs, shock_ys, actual_ys)
    end

    return NPointBowShockInterpolation(N, actual_ys, actual_xs)
end

# prefer methods to direct field access in case I can re-use some of these other methods later?

n_points(interp::NPointBowShockInterpolation) = interp.N
interp_x_knots(interp::NPointBowShockInterpolation) = interp.xs
interp_y_knots(interp::NPointBowShockInterpolation) = interp.ys

# TODO purge allocations here if they become a performance problem

function interpolation_points(interp)
    return map((x, y) -> SVector(x, y), interp_x_knots(interp), interp_y_knots(interp))
end

function interpolation_segments(interp)
    pts = interpolation_points(interp)
    p0 = SVector(pts[1][1], -pts[1][2])
    pushfirst!(pts, p0)
    return tuple.(@view(pts[begin:end-1]), @view(pts[begin+1:end]))
end

function _segment_normal(A, B)
    ℓ = PlanePolygons.Line(A, B - A)
    return normalize(left_normal(ℓ))
end

function interpolation_normals_at_points(interp)
    segments = interpolation_segments(interp)
    n_end = _segment_normal(last(segments)...)
    normals = map(@view(segments[begin:end-1]), @view(segments[begin+1:end])) do s1, s2
        n1 = _segment_normal(s1...)
        n2 = _segment_normal(s2...)
        return normalize(0.5 * (n1 + n2))
    end
    push!(normals, n_end)
end

function _interpolation_segment(interp, y)
    y0 = first(interp.ys)
    if y < y0
        return (SVector(first(interp.xs), -y0), SVector(first(interp.xs), y0))
    elseif y > last(interp.ys)
        return _interpolation_segment(interp, last(interp.ys))
    end

    Δy = step(interp.ys)
    n = min(interp.N, floor(Int, (y - y0) / Δy) + 1)
    return (SVector(interp.xs[n], interp.ys[n]), SVector(interp.xs[n+1], interp.ys[n+1]))
end

function _interpolation_normal(interp, y)
    return _segment_normal(_interpolation_segment(interp, y)...)
end

function bow_shock_point(interp, y)
    y★ = y < 0 ? -y : y
    p0, p1 = _interpolation_segment(interp, y★)
    dy★ = y★ - p0[2]
    s = dy★ / (p1[2] - p0[2])
    x★ = p0[1] + s * (p1[1] - p0[1])
    return SVector(x★, y < 0 ? -y★ : y★)
end

function bow_shock_normal(interp, y)
    y★ = y < 0 ? -y : y
    n = _interpolation_normal(interp, y★)
    if y < 0
        @reset n[2] = -1 * n[2]
    end
    return n
end

function shift_shock_interpolation_by_normals(interp, Δ)
    ns = interpolation_normals_at_points(interp)
    xs_shifted = map(interp_x_knots(interp), ns) do x, n
        return x + Δ * n[1]
    end
    ys_shifted = map(interp_y_knots(interp), ns) do y, n
        return y + Δ * n[2]
    end
    return NPointBowShockInterpolation(interp.N, ys_shifted, xs_shifted)
end

end
