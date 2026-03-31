module FullPseudoinversion

using LinearAlgebra

using Accessors
using DifferentiationInterface
using Graphs
using MetaGraphsNext
using OhMyThreads: tmap, tmapreduce
using StaticArrays
using Tullio

using Euler2D
using Euler2D: CellNeighboring, OTHER_QUADCELL, BOUNDARY_CONDITION, IS_PHANTOM
using Euler2D: shift_polygon_vertices_in_direction, fdiff_eps, select_middle
using Euler2D: all_cells_overlapping, total_mass_contained_by
using Euler2D: total_mass_contained_by_with_gradient_estimates
using Euler2D: fdiff_backend, estimate_∇u_in_rectangular_cell

using PlanePolygons

### POSTPROCESS A SIMULATION BY EXPANDING THE SYMMETRY OVER THE X-AXIS

# update the cell id and neighbors of a cell that is still in the old region
# id -> 2*id-1
function _update_cell_id_and_insert_into!(new_cells, old_cell)
    new_cell = @set old_cell.id = 2 * old_cell.id - 1
    @reset new_cell.neighbors = map(new_cell.neighbors) do (nbr, i)
        return nbr == OTHER_QUADCELL ? (nbr, 2 * i - 1) : (nbr, i)
    end
    return new_cells[new_cell.id] = new_cell
end

# update the cell id, neighbors, and values of a cell that's in the flipped region
# id -> 2*id
function _flip_cell_and_insert_into!(new_cells, old_cell)
    new_cell = @set old_cell.id = 2 * old_cell.id
    @reset new_cell.neighbors = map(new_cell.neighbors) do (nbr, i)
        return nbr == OTHER_QUADCELL ? (nbr, 2 * i) : (nbr, i)
    end
    old_south = new_cell.neighbors.south
    @reset new_cell.neighbors.south = new_cell.neighbors.north
    @reset new_cell.neighbors.north = old_south
    @reset new_cell.center = new_cell.center .* SVector(1, -1)
    flip = Diagonal(SVector(1, 1, -1, 1))
    @reset new_cell.u = flip * new_cell.u
    @reset new_cell.u̇ = flip * new_cell.u̇
    return new_cells[new_cell.id] = new_cell
end

function _update_cell_idx_field!(cells, cell_ids)
    for i in eachindex(IndexCartesian(), cell_ids)
        cell_ids[i] == 0 && continue
        cell = cells[cell_ids[i]]
        cells[cell_ids[i]] = @set cell.idx = i
    end
    return
end

function _expand_final_tstep_over_x_axis(sim)
    (Nx, Ny_old) = grid_size(sim)
    new_grid_size = (Nx, 2 * Ny_old)
    new_cell_ids = similar(sim.cell_ids, new_grid_size)

    known_region = @view new_cell_ids[:, (begin + Ny_old):end]
    for i in eachindex(sim.cell_ids, known_region)
        known_region[i] = 2 * sim.cell_ids[i] - (sim.cell_ids[i] != 0)
    end
    new_region = @view new_cell_ids[:, Ny_old:-1:begin]
    new_region .= 2 .* sim.cell_ids

    (bounds_x, (_, yR)) = sim.bounds
    new_bounds = (bounds_x, (-yR, yR))
    (t_last, cells) = nth_step(sim, n_tsteps(sim))
    new_cells = typeof(cells)()
    sizehint!(new_cells, 2 * length(cells))

    for (_, old_cell) in cells
        _update_cell_id_and_insert_into!(new_cells, old_cell)
        _flip_cell_and_insert_into!(new_cells, old_cell)
    end

    _update_cell_idx_field!(new_cells, new_cell_ids)
    foreach(@view(known_region[:, begin]), @view(new_region[:, begin])) do id_A, id_B
        (id_A == 0 || id_B == 0) && return
        cell_A = new_cells[id_A]
        cell_B = new_cells[id_B]
        new_cells[id_A] = @set cell_A.neighbors.south = (OTHER_QUADCELL, id_B)
        new_cells[id_B] = @set cell_B.neighbors.north = (OTHER_QUADCELL, id_A)
    end

    return CellBasedEulerSim(
        new_grid_size,
        1,
        new_bounds,
        [t_last],
        new_cell_ids,
        [new_cells],
    )
end

###

### SOME POLYGON STUFF
# To my future self: sorry

"""
    intersection_point_jacobian(point, poly1, poly2)

Get the Jacobian of the point (x, y) w.r.t the vertices of `poly1`,
provided (x, y) is the intersection of a pair of edges of `poly1` and `poly2`.
"""
function intersection_point_jacobian(point, poly1, poly2)
    N = num_vertices(poly1)
    J = zeros(eltype(point), (2, 2 * N))
    foreach(enumerate(zip(edge_starts(poly1), edge_ends(poly1)))) do (i, (p1, p2))
        if is_other_point_on_line(Line(p1, p2 - p1), point)
            for ell2 in
                Iterators.filter(ℓ -> is_other_point_on_line(ℓ, point), edge_lines(poly2))
                jac = DifferentiationInterface.jacobian(fdiff_backend, vcat(p1, p2)) do vals
                    q1 = vals[SVector(1, 2)]
                    q2 = vals[SVector(3, 4)]
                    return line_intersect(Line(q1, q2 - q1), ell2)
                end
                j = ((i + 1) % N) + 1
                J[(2 * i - 1):(2 * i)] = jac[SVector(1, 2)]
                J[(2 * j - 1):(2 * j)] = jac[SVector(3, 4)]
            end
        end
    end
    return J
end

function intersection_point_jacobian(
        point,
        poly1::SizedClockwiseOrientedPolygon{N, T},
        poly2,
    ) where {N, T}
    J = @SArray zeros(T, 2, 2 * N)
    for (i, (p1, p2)) in enumerate(zip(edge_starts(poly1), edge_ends(poly1)))
        if !is_other_point_on_line(Line(p1, p2 - p1), point)
            continue
        end
        for ℓ2 in Iterators.filter(ℓ -> is_other_point_on_line(ℓ, point), edge_lines(poly2))
            jac = DifferentiationInterface.jacobian(fdiff_backend, vcat(p1, p2)) do vals
                q1 = vals[SVector(1, 2)]
                q2 = vals[SVector(3, 4)]
                return line_intersect(Line(q1, q2 - q1), ℓ2)
            end
            j = ((i + 1) % N) + 1
            @reset J[(2 * i - 1):(2 * i)] = jac[SVector(1, 2)]
            @reset J[(2 * j - 1):(2 * j)] = jac[SVector(3, 4)]
        end
    end
    return J
end

function intersection_point_gradient(
        point,
        poly1::SClosedPolygon{N, T},
        shift_dirs,
        poly2,
    ) where {N, T}
    @assert N == length(shift_dirs)
    # just use a regular array here?
    J = @MMatrix zeros(T, 2, N)
    # A and B are on i-th edge of poly1
    for (i, (A, B)) in enumerate(zip(edge_starts(poly1), edge_ends(poly1), shift_dirs))
        ℓ_AB = Line(A, B - A)
        if !is_other_point_on_line(ℓ_AB, point)
            continue
        end
        # the jacobian actually ends up being either all zeros on only having two nonzero columns?
        for ℓ_BC in edge_lines(poly2)
            if !is_other_point_on_line(ℓ_BC, point)
                continue
            end
            # which pair of shifts i,j correspond to A, B
            j = ((i + 1) % N) + 1
            jac_local = DifferentiationInterface.jacobian(
                fdiff_backend,
                @SVector(zeros(T, 2)),
                Constant(A),
                Constant(B),
                Constant(shift_dirs[i]),
                Constant(shift_dirs[j]),
                Constant(ℓ_BC),
            ) do s, pA, pB, dirA, dirB, ℓ_BC_noshift
                A_shifted = pA + s[1] * dirA
                B_shifted = pB + s[2] * dirB
                ℓ_AB_shifted = Line(A_shifted, B_shifted - A_shifted)
                return line_intersect(ℓ_AB_shifted, ℓ_BC_noshift)
            end
            J[:, i] = jac_local[:, 1]
            J[:, j] = jac_local[:, 2]
        end
    end
    return J
end

# gets the jacobian of the intersection area w.r.t. the first argument
# allocates like hell, but... it works and isn't performance critical.?
function intersection_area_jacobian(flat_poly1, poly2)
    grad1 = zero(flat_poly1)
    for i in eachindex(flat_poly1)
        h = fdiff_eps(flat_poly1[i])
        in1 = @set flat_poly1[i] += h
        in2 = @set flat_poly1[i] -= h
        out1 = poly_area(poly_intersection(in1, poly2))
        out2 = poly_area(poly_intersection(in2, poly2))
        @reset grad1[i] = (out1 - out2) / (2 * h)
    end
    return grad1
end

"""
    intersection_area_gradient(poly1, s_dirs, poly2)

Compute the gradient of the intersection area of `poly1` and `poly2` with respect to
shift magnitudes in the given shift directions. 

We assume that `poly1` is what is shifted and that `poly2` remains constant.

Uses finite differences. There is almost certainly an AD-based solution here...
however, we are interested in some amount of "smoothing", in case the intersection area is not very large.
"""
function intersection_area_gradient(poly1::SClosedPolygon{N, T}, s_dirs, poly2) where {N, T}
    grad = @SVector zeros(T, N)
    h = fdiff_eps(zero(T))
    s = @SVector zeros(T, N)
    for i in eachindex(s, grad)
        s_cur = @set s[i] -= h
        poly1_minus_h =
            SClosedPolygon(shift_polygon_vertices_in_direction(poly1, s_dirs, s_cur))
        s_cur = @set s[i] += h
        poly1_plus_h =
            SClosedPolygon(shift_polygon_vertices_in_direction(poly1, s_dirs, s_cur))
        # maybe save a buffer here?
        # made a change so that DifferentiationInterface could handle some things
        # need to assert that we do have SClosedPolygons here
        A_minus = poly_area(poly_intersection(poly1_minus_h, poly2))
        A_plus = poly_area(poly_intersection(poly1_plus_h, poly2))
        @reset grad[i] = (A_plus - A_minus) / (2 * h)
    end
    return grad
end

## END POLYGON STUFF

struct PseudoinversionCell{T, NSEEDS, NTANGENTS} <: FVMCell{T}
    id::Int
    boundary::SClosedPolygon{4, T}
    vertex_shift_dirs::SVector{4, SVector{2, T}}
    u::SVector{4, T}
    u̇::SMatrix{4, NSEEDS, T, NTANGENTS}
    du_ds::SMatrix{4, 4, T, 16}
end

function PseudoinversionCell{T, NS}(id, boundary, shift_dirs) where {T, NS}
    return PseudoinversionCell{T, NS, 4 * NS}(
        id,
        boundary,
        shift_dirs,
        @SVector(zeros(T, 4)),
        @SMatrix(zeros(T, 4, NS)),
        @SMatrix(zeros(T, 4, 4)),
    )
end

Euler2D.cell_boundary_polygon(cell::PseudoinversionCell) = cell.boundary

using Euler2D.BowShockInterpolation: n_points, x_knots, y_knots
using Euler2D.BowShockInterpolation: interpolation_segments, interpolation_normals_at_points

"""
    pseudoinversion_cell_polys(bow_shock_interp, L)

Build the set of polygons for pseudoinversion with tangent-to-shock side lengths `L`.
"""
function pseudoinversion_cell_polys(segments, corner_normals, L)
    out = SClosedPolygon{4, Float64}[]
    sizehint!(out, 4 * length(segments))
    for i in eachindex(segments)
        n1 = corner_normals[begin + i - 1]
        n2 = corner_normals[begin + i]
        A, B = segments[i]
        Qi_LL = SClosedPolygon(A - 2 * L * n1, B - 2 * L * n2, B - L * n2, A - L * n1)
        Qi_L = SClosedPolygon(A - L * n1, B - L * n2, B, A)
        Qi = SClosedPolygon(A, B, B + L * n2, A + L * n1)
        Qi_R = SClosedPolygon(A + L * n1, B + L * n2, B + 2 * L * n2, A + 2 * L * n1)
        push!(out, Qi_LL, Qi_L, Qi, Qi_R)
    end
    return out
end

# Page 23 of my red notebook;
function _du_ds_via_product_rule(
        u_j,
        ∇x_u_j,
        x_center_j,
        sim_cell_boundary,
        pinv_cell_boundary,
        pinv_shift_dirs,
    )
    # intersection between the pinv and simulation cells
    p_isect = poly_intersection(pinv_cell_boundary, sim_cell_boundary)
    # area of that intersection
    A_isect = poly_area(p_isect)
    # and its dependence on s
    ∇s_A_isect = #@SVector zeros(eltype(u_j), length(pinv_shift_dirs))
        intersection_area_gradient(pinv_cell_boundary, pinv_shift_dirs, sim_cell_boundary)

    # Approximate value of the linear interpolation of u in the intersection...
    u_isect = inv(num_vertices(p_isect)) * sum(edge_starts(p_isect)) do pt_l
        return u_j + ∇x_u_j * (pt_l - x_center_j)
    end
    # dependence of u_isect on the shifts s
    du_isect_ds = # @SMatrix zeros(eltype(u_j), 4, length(pinv_shift_dirs))
        inv(num_vertices(p_isect)) *
        ∇x_u_j *
        sum(edge_starts(p_isect)) do pt_l
        ∇s_pt_l = intersection_point_gradient(
            pt_l,
            pinv_cell_boundary,
            pinv_shift_dirs,
            sim_cell_boundary,
        )
        # factor out ∇u, since it's constant (linear approximation)
        return ∇s_pt_l
    end
    # A is a scalar -> dAds is a row vector
    # (u in R^4) * (dA' in R^[1 x n_s]) * Δs -> something in R^4
    # plus
    # (A in R) * (du_ds in R^[4 x n_s]) -> something in R^4
    # sizes check out
    return u_isect * ∇s_A_isect' + A_isect * du_isect_ds
end

function compute_pseudoinversion_cell_contents(
        pinv_cell::PseudoinversionCell{T, NS, NT},
        sim::CellBasedEulerSim{T, TangentQuadCell{T, NS, NT}},
        tstep,
        boundary_conditions,
        gas,
    ) where {T, NS, NT}
    pinv_cell_boundary = cell_boundary_polygon(pinv_cell)
    A, dAds = DifferentiationInterface.value_and_gradient(
        fdiff_backend,
        @SVector(zeros(T, 4)),
        Constant(pinv_cell_boundary),
        Constant(pinv_cell.vertex_shift_dirs),
    ) do s, poly, ns
        return poly_area(vcat(shift_polygon_vertices_in_direction(poly, ns, s)...))
    end
    _, cells = nth_step(sim, tstep)
    min_cell_size = minimum_cell_size(sim)
    overlapped = all_cells_overlapping(pinv_cell_boundary, sim; padding = min_cell_size)
    U_tot, Udot_tot = total_mass_contained_by_with_gradient_estimates(
        pinv_cell_boundary,
        sim,
        tstep,
        boundary_conditions,
        gas;
        poly_bbox_padding = min_cell_size,
    )
    # compute the first term of the product rule expansion of
    # partial u partial s for the pinv cell
    # the entire derivation is on page 22 of my red notebook
    # eqn is marked with **
    # Interested in the first sum term inside the big brackets
    # namely Σ_j ∂_s_k * ∬_(Cj ∩ Qi) u(x) dA
    dU_tot_ds = mapreduce(+, overlapped) do id
        ∇_x_u = estimate_∇u_in_rectangular_cell(cells[id], sim, tstep, boundary_conditions, gas)
        return _du_ds_via_product_rule(
            cells[id].u,
            ∇_x_u,
            cell_center(cells[id]),
            cell_boundary_polygon(cells[id]),
            pinv_cell_boundary,
            pinv_cell.vertex_shift_dirs,
        )
    end
    du_ds = inv(A^2) * (A * dU_tot_ds - U_tot * dAds')
    return inv(A) * U_tot, inv(A) * Udot_tot, du_ds
end

function make_pseudoinversion_cells(
        sim::CellBasedEulerSim{T, <:TangentQuadCell{T, NS, NP}},
        bow_shock_interp,
        L_tangent,
        bcs,
        gas,
    ) where {T, NS, NP}
    segments = interpolation_segments(bow_shock_interp)
    corner_normals = collect(interpolation_normals_at_points(bow_shock_interp))
    pushfirst!(corner_normals, first(corner_normals) .* SVector(1, -1))
    normals_S = @view corner_normals[begin:(end - 1)]
    normals_N = @view corner_normals[(begin + 1):end]
    polys = pseudoinversion_cell_polys(segments, corner_normals, L_tangent)

    pinv_cells = PseudoinversionCell{T, NS, NP}[]
    sizehint!(pinv_cells, length(polys))
    polys_view = reshape(polys, (4, n_points(bow_shock_interp)))
    for i in eachindex(normals_S, normals_N)
        n1 = normals_S[i]
        n2 = normals_N[i]
        for j in axes(polys_view, 1)
            poly = polys_view[j, i]
            shift_directions = if j % 4 == 2
                # cell is to the "left" of the shock
                # so we set its shift directions like this
                SVector(@SVector(zeros(T, 2)), @SVector(zeros(T, 2)), n2, n1)
            elseif j % 4 == 3
                # cell is to the "right" of the shock
                # so we set shift directions on the other two vertices
                SVector(n1, n2, @SVector(zeros(T, 2)), @SVector(zeros(T, 2)))
            else
                # cell is not on the shock so we set its shift direction to be all zeros
                SVector(
                    @SVector(zeros(T, 2)),
                    @SVector(zeros(T, 2)),
                    @SVector(zeros(T, 2)),
                    @SVector(zeros(T, 2)),
                )
            end
            push!(pinv_cells, PseudoinversionCell{T, NS}(i, poly, shift_directions))
        end
    end

    for i in eachindex(pinv_cells)
        pinv_cell = pinv_cells[i]
        contents = compute_pseudoinversion_cell_contents(pinv_cell, sim, 1, bcs, gas)
        @reset pinv_cell.u = contents[1]
        @reset pinv_cell.u̇ = contents[2]
        @reset pinv_cell.du_ds = contents[3]
        pinv_cells[i] = pinv_cell
    end
    return pinv_cells
end

using Euler2D.ExactRiemannSolver: u_riemann

function _boundary_integral(cell_id, pinv_cells, gas)
    @assert cell_id % 4 == 3 || cell_id % 4 == 2
    p = cell_boundary_polygon(pinv_cells[cell_id])
    normals_p = outward_edge_normals(p)
    edge_lengths = map((a, b) -> norm(b - a), edge_starts(p), edge_ends(p))
    u_west = u_riemann(pinv_cells[cell_id - 1].u, pinv_cells[cell_id].u, -normals_p[1], gas)
    u_north = u_riemann(pinv_cells[cell_id].u, pinv_cells[cell_id + 4].u, normals_p[2], gas)
    u_east = u_riemann(pinv_cells[cell_id].u, pinv_cells[cell_id + 4].u, normals_p[3], gas)
    u_south = u_riemann(pinv_cells[cell_id - 3].u, pinv_cells[cell_id].u, -normals_p[4], gas)
    return sum(
        zip(edge_lengths, (u_west, u_north, u_east, u_south), normals_p),
    ) do (L, u, n)
        L * F_euler(u, gas) * n
    end
end
end
