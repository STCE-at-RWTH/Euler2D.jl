# IMPLEMENTATION OF
# Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows  
# Takeshi R. Fujimoto ∗, Taro Kawasaki 1, Keiichi Kitamura
# Journal of Computational Physics, 396, pp. 264 - 279
#

module CannyShockSensor

using Euler2D
using Euler2D: CellBasedEulerSim, select_middle

using LinearAlgebra
using ShockwaveProperties
using StaticArrays
using Tullio

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

"""
    mark_maxima(dP2, θ)

Performs edge thinning on dP2 using gradient directions θ.
"""
function mark_maxima(dP2, θ_ij)
    candidates = fill(false, size(dP2) .- 2)
    for i ∈ eachindex(candidates)
        dP2_view = @view dP2[i:i+CartesianIndex(2, 2)]
        candidates[i] = is_edge_candidate(dP2_view, θ_ij[i+CartesianIndex(1, 1)])
    end
end

function _shock_velocity(θij, u1, u2, gas)
    h1 = dimensionless_total_enthalpy(u1, gas)
    h2 = dimensionless_total_enthalpy(u2, gas)
    # q = dot(v,v)
    v1 = select_middle(u1) / u1[1]
    v2 = select_middle(u2) / u2[1]
    Δv = v1 - v2
    A = 2 * h1 + v1 ⋅ v1
    B = 2 * h2 + v2 ⋅ v2
    vs_x = (A - B) / (2 * (Δv[1] + tan(θij) * Δv[2]))
    vs_y = (A - B) / (2 * (cot(θij) * Δv[1] + Δv[2]))
    return SVector(vs_x, vs_y)
end

"""
    rh_errors_shock_frame(θij, u1, u2, gas)

Compute the Mach number jump error and a smoothness criterion between two states `u1` and `u2`.

1. Translate the states `u1` and `u2` into a coordinate system that moves with the shock.
`u1` is upstream, `u2` is downstream. In the bow-shock problem, we want to assert that there is
a pressure increase across the shock (upstream -> downstream).
2. Estimate the change across that shock. Note that `a` does not change under a velocity transformation
3. Return the error between the computed value and the simulated value, as well as an estimate of the "strength" of the shock.
"""
function rh_errors_shock_frame(θij, u1, u2, gas)
    @assert dimensionless_pressure(u1, gas) ≤ dimensionless_pressure(u2, gas)
    v1 = select_middle(u1) / u1[1]
    v2 = select_middle(u2) / u2[1]
    v_s = _shock_velocity(θij, u1, u2, gas)
    m1 = (v1 - v_s) ./ dimensionless_speed_of_sound(u1, gas)
    m2 = (v2 - v_s) ./ dimensionless_speed_of_sound(u2, gas)
    n̂ = normalize(v_s)
    m1_normal = abs(m1 ⋅ n̂)
    m2_normal = abs(m2 ⋅ n̂)
    if m1_normal > m2_normal
        @warn "Shock assumptions violated?" s = norm(v_s) m1_normal m2_normal
    end
    m2sqr = (2 + (gas.γ - 1) * m1_normal^2) / (2 * gas.γ * m1_normal^2 - (gas.γ - 1))
    m2_predicted = sqrt(m2sqr)
    mach_jump_error = abs(m2_normal - m2_predicted) / m2_normal
    smoothness_error = abs((m2_normal - m1_normal)) / m1_normal
    return mach_jump_error, smoothness_error
end

#assumes stationary shock "edge"
function rh_error_lab_frame(cell_front, cell_behind, θ, gas)
    m1 = dimensionless_mach_number(cell_front.u, gas)
    m2 = dimensionless_mach_number(cell_behind.u, gas)
    dir = sincos(θ)
    n̂ = SVector(dir[2], dir[1])
    m_ratio = ShockwaveProperties.shock_normal_mach_ratio(m1, n̂, gas)
    m1_norm = abs(m1 ⋅ n̂)
    m2_norm_rh = m1_norm * m_ratio
    m2_norm_sim = abs(m2 ⋅ n̂)
    return (abs(m2_norm_rh - m2_norm_sim) / m2_norm_sim, abs(m1_norm / m2_norm_sim - 1))
end

function is_candidate_cell_shock(
    cell_front,
    cell_back,
    θ_disc,
    gas,
    rh_thold,
    smoothness_thold,
)
    try
        rh_err, sim_err = rh_error_lab_frame(cell_front, cell_back, θ_disc, gas)
        if rh_err > rh_thold || sim_err < smoothness_thold
            return false
        end
    catch possible_domainerr
        if possible_domainerr isa DomainError
            return false
        else
            rethrow()
        end
    end
    return true
end

struct ShockSensorInfo
    candidates::Matrix{Bool}
    domainerred::Matrix{Bool}
    n_candidate_cells::Int
    n_erred::Int
    n_thinned::Int
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
    @show size(pfield), grid_size(sim)
    pfield_nopad = @view pfield[begin+2:end-2, begin+2:end-2]
    @show size(pfield_nopad)
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
    dP2_nopad = @view dPdy[begin+1:end-1, begin+1:end-1]
    candidates = Matrix{Bool}(undef, grid_size(sim))
    for i ∈ eachindex(IndexCartesian(), candidates, dPdx_nopad, dPdy_nopad)
        window = i:(i+CartesianIndex(2, 2))
        θ = atan(dPdy_nopad[i], dPdx_nopad[i])
        candidates[i] = is_edge_candidate(@view(dP2[window]), θ)
    end
    n_marked_edges = count(candidates)
    n_too_smooth = 0
    n_no_shock = 0
    erred = fill(false, grid_size(sim))

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
        u_A = cells[sim.cell_ids[i+Δi]].u
        u_B = cells[sim.cell_ids[i-Δi]].u
        try
            errs = if dimensionless_pressure(u_A, gas) < dimensionless_pressure(u_B, gas)
                rh_errors_shock_frame(θij, u_A, u_B, gas)
            else
                rh_errors_shock_frame(θij, u_B, u_A, gas)
            end
            if errs[1] > rh_rel_error_max
                n_no_shock += 1
                candidates[i] = false
            elseif errs[2] <= continuous_variation_thold
                n_too_smooth += 1
                candidates[i] = false
            end
        catch err
            if !(err isa DomainError)
                rethrow()
            else
                candidates[i] = false
                erred[i] = true
            end
        end
    end

    return ShockSensorInfo(
        candidates,
        erred,
        count(candidates),
        count(erred),
        n_marked_edges,
        n_too_smooth,
        n_no_shock,
    )
end

end

function shock_cells(sim, n, shock_field)
    sort(
        reduce(
            vcat,
            filter(!isnothing, map(enumerate(eachcol(shock_field))) do (j, col)
                i = findfirst(col)
                isnothing(i) && return nothing
                id = @view(sim.cell_ids[2:end-1, 2:end-1])[i, j]
                return sim.cells[n][id]
            end),
        );
        lt = (a, b) -> a.center[2] < b.center[2],
    )
end

function shock_points(sim::CellBasedEulerSim{T,C}, n, shock_field) where {T,C}
    sp = shock_cells(sim, n, shock_field)
    res = Matrix{T}(undef, (length(sp), 2))
    for i ∈ eachindex(sp)
        res[i, :] = sp[i].center
    end
    #sort!(sp; lt=(a, b) -> a[2] < b[2])
    return res
end
