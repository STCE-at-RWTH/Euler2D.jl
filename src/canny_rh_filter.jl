# IMPLEMENTATION OF
# Canny-Edge-Detection/Rankine-Hugoniot-conditions unified shock sensor for inviscid and viscous flows  
# Takeshi R. Fujimoto ∗, Taro Kawasaki 1, Keiichi Kitamura
# Journal of Computational Physics, 396, pp. 264 - 279

_diff_op(T) = SVector{3,T}(one(T), zero(T), -one(T))
_avg_op(T) = SVector{3,T}(one(T), 2 * one(T), one(T))

function convolve_sobel(matrix::AbstractMatrix{T}) where {T}
    Gy = _avg_op(T) * _diff_op(T)'
    Gx = _diff_op(T) * _avg_op(T)'
    @show Gx, Gy
    new_size = size(matrix) .- 2
    outX = similar(matrix, new_size)
    outY = similar(matrix, new_size)
    for i ∈ eachindex(IndexCartesian(), outX, outY)
        view_range = i:(i+CartesianIndex(2, 2))
        outX[i] = Gx ⋅ @view(matrix[view_range])
        outY[i] = Gy ⋅ @view(matrix[view_range])
    end
    return outX, outY
end

gradient_magnitude2(Gx, Gy) = Gx .^ 2 .+ Gy .^ 2
gradient_direction(Gx, Gy) = atan(Gy ./ Gx)

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

function mark_edge_candidate(dP2_view, Gx, Gy)
    grid_theta = gradient_grid_direction(atan(Gy, Gx))
    idx = CartesianIndex(2, 2)
    return dP2_view[idx+grid_theta] < dP2_view[idx] &&
           dP2_view[idx-grid_theta] < dP2_view[idx]
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

function relative_rankine_hugoniot_error(u1, u2, θ, gas)
    m1 = dimensionless_mach_number(u1, gas)
    m2 = dimensionless_mach_number(u2, gas)
    dir = sincos(θ)
    n̂ = SVector(dir[2], dir[1])
    m_ratio = ShockwaveProperties.shock_normal_mach_ratio(m1, n̂, gas)
    m1_n = abs(m1 ⋅ n̂)
    m2_norm_theo = m1_n * m_ratio
    m2_norm_sim = abs(m2 ⋅ n̂)
    return abs(m2_norm_theo - m2_norm_sim) / m2_norm_sim
end

function strong_shock_error(u1, u2, θ, gas)
end

function find_shock_in_timestep(
    sim::CellBasedEulerSim{T,C},
    t,
    gas;
    rh_rel_error_max = 0.5,
    continuous_variation_thold = 0.04,
) where {T,C}
    # TODO really gotta figure out how to deal with nothings or missings in this matrix
    pfield = map(p -> isnothing(p) ? 0.0 : p, pressure_field(sim, t, gas))
    Gx, Gy = convolve_sobel(pfield)
    dP2 = gradient_magnitude2(Gx, Gy)
    edge_candidates = Array{Bool,2}(undef, size(dP2) .- 2)
    window_size = CartesianIndex(2, 2)
    for i ∈ eachindex(IndexCartesian(), edge_candidates)
        edge_candidates[i] = mark_edge_candidate(
            @view(dP2[i:i+window_size]),
            Gx[i+CartesianIndex(1, 1)],
            Gy[i+CartesianIndex(1, 1)],
        )
    end
    @info "Number of candidates..." n_candidates = sum(edge_candidates)
    Gx_overlay = @view(Gx[2:end-1, 2:end-1])
    Gy_overlay = @view(Gy[2:end-1, 2:end-1])
    id_overlay = @view(sim.cell_ids[3:end-2, 3:end-2])
    for j ∈ eachindex(IndexCartesian(), edge_candidates, Gx_overlay, Gy_overlay, id_overlay)
        i = j + CartesianIndex(2, 2)
        if id_overlay[j] > 0 && edge_candidates[j]
            θ = atan(Gy_overlay[j], Gx_overlay[j])
            θ_disc = discretize_gradient_direction(θ)
            θ_grid = gradient_grid_direction(θ_disc)
            # gradient points in direction of steepest increase...
            # cell in "front" of shock should be opposite the gradient?
            id_front = sim.cell_ids[i-θ_grid]
            id_back = sim.cell_ids[i+θ_grid]
            if id_front == 0 || id_back == 0
                edge_candidates[j] = false
                continue
            end

            cell_front = sim.cells[t][id_front]
            cell_back = sim.cells[t][id_back]
            try
                rh_err, sim_err = rh_error_lab_frame(cell_front, cell_back, θ_disc, gas)
                @show rh_err, sim_err
                if rh_err > rh_rel_error_max || sim_err < continuous_variation_thold
                    # discard edge candidate
                    edge_candidates[j] = false
                end
            catch de
                #@warn "Cell shock comparison caused DomainError" j θ_grid
                edge_candidates[j] = false
            end
        else
            edge_candidates[j] = false
        end
    end
    @info "Number of candidates after double thresholding..." n_candidates =
        sum(edge_candidates)
    return edge_candidates
end